from cProfile import label
from fileinput import filename
import os
import cv2
import sys
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from adabelief_pytorch import AdaBelief
from torch_poly_lr_decay import PolynomialLRDecay

import models
import attention as att
import label as lb
from datasets.cityscapes import Cityscapes

from torch.utils.tensorboard import SummaryWriter


def displayImage(imgList, filename="test.png"):
    fig = plt.figure(figsize=(15, 15))

    nColuns = 3
    nLines = int(np.ceil(len(imgList) / 3.0))

    for index, img in enumerate(imgList):
        fig.add_subplot(nLines, nColuns, index + 1)
        plt.imshow(img["img"], interpolation="bilinear")
        plt.title(img["title"])

    # plt.show()
    plt.savefig(filename)


def iou_coef(pred, labels):
    smooth = 0.01

    intersection = np.sum(np.abs(labels[0:19] * pred))
    union = np.sum(labels[0:19]) + np.sum(pred) - intersection

    iou = np.mean((intersection + smooth) / (union + smooth))

    return iou


def ct_loss(out, target):
    return (-(out + 1e-5).log() * target[:, 0:19]).sum(dim=1).mean()


def trainTrunk(args, model, trainGen, valGen, device):

    ft_flag = args.ft_flag
    learning_rate = args.learning_rate

    model.to(device)

    freeze_flag = False
    if ft_flag:
        model.freezeTrunk()
        optimizer = AdaBelief(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        freeze_flag = True
    else:
        optimizer = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-16, betas=(0.9, 0.999), weight_decay=1e-4)
        lr_schedule = PolynomialLRDecay(optimizer, max_decay_steps=600000, end_learning_rate=0.0005, power=2.0)

    log_dir = None if (args.metrics_path == "") else "runs/" + args.metrics_path
    writer = SummaryWriter(log_dir=log_dir)

    maxIOU = -1
    epochs = args.num_epochs

    for e in tqdm(range(epochs)):
        model.train()
        for l_image, l_label in tqdm(trainGen, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):

            image, labels = l_image.to(device), l_label.to(device)
            print(image.shape)

            optimizer.zero_grad()
            trunk, out = model(image)
            loss = ct_loss(out, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            if ft_flag and e > 0.6 * epochs and freeze_flag:
                model.unfreezeTrunk()
                optimizer.add_param_group({"params": model.new_model.parameters()})
                freeze_flag = False

            if not ft_flag:
                lr_schedule.step()

            break

        # -----------validation------------
        mIOUsum = 0
        model.eval()
        for l_image, l_label in tqdm(valGen, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):
            image, labels = l_image.to(device), l_label.to(device)
            trunk, out = model(image)
            mIOU = iou_coef(out.cpu().detach().numpy()[0], labels.cpu().detach().numpy()[0])
            mIOUsum += mIOU
            # break

        mIOUsum = float(mIOUsum / len(valGen))
        print("\n\n")
        print(mIOUsum)
        print("\n\n")
        writer.add_scalar("Loss/train", loss, e)
        writer.add_scalar("Accuracy/val", mIOUsum, e)

        # save model
        if mIOUsum > maxIOU:
            maxIOU = mIOUsum
            if (e > 0.4 * epochs or args.ft_flag) and args.save_model_path != "":
                torch.save(model.state_dict(), args.save_model_path)

    writer.flush()
    writer.close()

    print("Max mIOU: ", maxIOU)

    # display image
    model.eval()
    for l_image, l_label in valGen:
        image, labels = l_image.to(device), l_label.to(device)
        imgList = []
        imgList.append({"title": "Original", "img": l_image[0].permute(1, 2, 0)})

        trunk, out = model(image)
        c_pred = out.cpu().detach().numpy()[0]  # .transpose((1, 2, 0))

        c_pred = lb.cityscapes_pallete[np.argmax(c_pred, axis=0), :]
        c_label = lb.cityscapes_pallete[np.argmax(l_label[0], axis=0), :]

        imgList.append({"title": "Color pred", "img": c_pred})
        imgList.append({"title": "Color label", "img": c_label})
        displayImage(imgList)
        break


def trainAtt(args, trunkModel, attModel, trainGen, valGen, device):

    trunkModel.eval()
    trunkModel.to(device)
    attModel.to(device)

    optimizer = AdaBelief(attModel.parameters(), lr=0.01, eps=1e-16, betas=(0.9, 0.999), weight_decay=1e-4)
    lr_schedule = PolynomialLRDecay(optimizer, max_decay_steps=600000, end_learning_rate=0.0005, power=2.0)

    log_dir = None if (args.metrics_path == "") else "runs/" + args.metrics_path
    writer = SummaryWriter(log_dir=log_dir)

    maxIoU = -1
    epochs = args.num_epochs

    for e in tqdm(range(epochs)):
        attModel.train()
        for l_imageL, l_imageH, l_label in tqdm(trainGen, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):

            imageL, imageH, labels = l_imageL.to(device), l_imageH.to(device), l_label.to(device)

            optimizer.zero_grad()
            trunk, predL = trunkModel(imageL)
            _, predH = trunkModel(imageH)

            attMask, out = attModel(trunk, predL, predH)

            loss = ct_loss(out, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            lr_schedule.step()

        # ----------- validation ------------ #
        mIoU_sum = 0
        attModel.eval()

        for l_imageL,l_imageH, l_label in tqdm(valGen, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):
            imageL, imageH, labels = l_imageL.to(device), l_imageH.to(device), l_label.to(device)

            # Forward step, Get predictions and trunk features
            trunk, predL = trunkModel(imageL)
            _, predH = trunkModel(imageH)

            attMask, out = attModel(trunk, predL.detach(), predH.detach())

            # Calculate mIoU
            mIoU = iou_coef(out.cpu().detach().numpy()[0], labels.cpu().detach().numpy()[0])
            mIoU_sum += mIoU

        mIoU_sum = float(mIoU_sum/len(valGen))
        writer.add_scalar('Loss/train', loss, e)
        writer.add_scalar('Accuracy/val', mIoU_sum, e)
        
        # save model
        if mIoU_sum > maxIoU:
            maxIoU = mIoU_sum
            if (e > 0.4 * epochs or args.ft_flag) and args.save_model_path != "":
                torch.save(attModel.state_dict(), args.save_model_path)

    # display image
    attModel.eval()
    for l_imageL, l_imageH, l_label in tqdm(valGen, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):
        imageL, imageH, labels = l_imageL.to(device), l_imageH.to(device), l_label.to(device)

        trunk, predL = trunkModel(imageL)
        _, predH = trunkModel(imageH)

        attMask, out = attModel(trunk, predL, predH)
        imgList = []
        imgList.append({"title": "Original", "img": imageL.cpu()[0].permute(1, 2, 0)})
        c_predL = predL.cpu().detach().numpy()[0]
        c_predH = predH.cpu().detach().numpy()[0]
        c_pred = out.cpu().detach().numpy()[0]

        c_predL = lb.cityscapes_pallete[np.argmax(c_predL, axis=0), :]
        c_predH = lb.cityscapes_pallete[np.argmax(c_predH, axis=0), :]
        c_pred = lb.cityscapes_pallete[np.argmax(c_pred, axis=0), :]
        c_label = lb.cityscapes_pallete[np.argmax(l_label[0], axis=0), :]

        imgList.append({"title": "Color predL", "img": c_predL})
        imgList.append({"title": "Color predH", "img": c_predH})
        imgList.append({"title": "attMask", "img": attMask.cpu().detach()[0].permute(1, 2, 0).numpy()})
        imgList.append({"title": "Color pred", "img": c_pred})
        imgList.append({"title": "Color label", "img": c_label})
        displayImage(imgList, filename="att.png")
        break

    writer.flush()
    writer.close()

    print("Max mIoU: ", maxIoU)
