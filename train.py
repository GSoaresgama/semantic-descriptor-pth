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

def displayImage(imgList, filename = "test.png"):
    fig = plt.figure(figsize=(15, 15))

    nColuns = 3
    nLines =  int(np.ceil(len(imgList)/3.0))

    for index, img in enumerate(imgList):
        fig.add_subplot(nLines, nColuns, index + 1)
        plt.imshow(img['img'], interpolation='bilinear')
        plt.title(img['title'])

    # plt.show()
    plt.savefig(filename)

def iou_coef(pred, labels):
    smooth = 0.01
    intersection = np.sum(np.abs(labels[0:18]*pred[0:18]))
    #print("intersection: ", intersection)
    # intersection = np.sum(np.abs(labels*pred))
    union = np.sum(labels[0:18]) + np.sum(pred[0:18]) - intersection
    #print("union: ", union)
    iou = np.mean((intersection+smooth)/(union+smooth))
    return iou

def ct_loss(out, target):
    #print("out shape:", out.shape)
    #print("target shape:", target.shape)
    loss = (-(out+1e-5).log() * target)[:,0:18].sum(dim=1).mean()
    return loss 


def trainTrunk(args, model, trainDataset, valDataset):
    pass


def trainAtt(args, trunkModel, attModel, trainGen, valGen, device):
    
    trunkModel.eval()
    trunkModel.to(device)
    attModel.to(device)

    optimizer = AdaBelief(attModel.parameters(), lr=0.00001, eps=1e-16, betas=(0.9,0.999))
    #lr_schedule = PolynomialLRDecay(optimizer, max_decay_steps=600000, end_learning_rate=0.00001, power=2.0)

    log_dir = None if(args.metrics_path == "") else "runs/" + args.metrics_path
    writer = SummaryWriter(log_dir=log_dir)

    maxIOU = -1
    epochs = args.num_epochs
    
    for e in tqdm(range(epochs)):
        for l_imageL, l_imageH, l_label in tqdm(trainGen, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):

            imageL, imageH, labels = l_imageL.to(device), l_imageH.to(device), l_label.to(device)

            optimizer.zero_grad()
            trunk, predL = trunkModel(imageL)
            _, predH = trunkModel(imageH)

            attMask, out = attModel(trunk, predL, predH)

            loss = ct_loss(out, labels)
            
            #loss = criterion(out, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            #lr_schedule.step()
            # break
        #-----------validation------------
        mIOUsum = 0
        for l_imageL,l_imageH, l_label in tqdm(valGen, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            imageL, imageH, labels = l_imageL.to(device), l_imageH.to(device), l_label.to(device)
            trunk, predL = trunkModel(imageL)
            _, predH = trunkModel(imageH)

            attMask, out = attModel(trunk, predL, predH)
            #print(out.shape)
            #print(out.cpu().detach().numpy().shape)
            mIOU = iou_coef(out.cpu().detach().numpy()[0], labels.cpu().detach().numpy()[0])
            mIOUsum += mIOU
            # break

        mIOUsum = float(mIOUsum/len(valGen))
        print("\n\n")
        print(mIOUsum)
        print("\n\n")
        writer.add_scalar('Loss/train', loss, e)
        writer.add_scalar('Accuracy/val', mIOUsum, e)
        
        #save model
        if mIOUsum > maxIOU:
            maxIOU = mIOUsum
            if e>-0.4*epochs and args.save_model_path != "":
                torch.save(attModel.state_dict(), args.save_model_path)
    
    for l_imageL,l_imageH, l_label in tqdm(valGen, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        imageL, imageH, labels = l_imageL.to(device), l_imageH.to(device), l_label.to(device)
        # print("imageL.shape",imageL.shape)

        trunk, predL = trunkModel(imageL)
        _, predH = trunkModel(imageH)

        attMask, out = attModel(trunk, predL, predH)
        imgList = [{'title': 'Original', 'img': imageL.cpu()[0].permute(1, 2, 0)},
                   {'title': 'Original', 'img': imageL.cpu()[0].permute(1, 2, 0)}]
        c_predL = predL.cpu().detach().numpy()[0]
        c_predH = predH.cpu().detach().numpy()[0]
        c_pred = out.cpu().detach().numpy()[0]
        # print(c_pred.shape)

        c_predL = lb.cityscapes_pallete[np.argmax(c_predL, axis=0), :]  
        c_predH = lb.cityscapes_pallete[np.argmax(c_predH, axis=0), :]  
        c_pred = lb.cityscapes_pallete[np.argmax(c_pred, axis=0), :]  
        c_label = lb.cityscapes_pallete[np.argmax(l_label[0], axis=0), :]  
        
        imgList.append({'title' : 'Color predL', 'img' : c_predL})
        imgList.append({'title' : 'Color predH', 'img' : c_predH})
        imgList.append({'title' : 'attMask', 'img' : attMask.cpu().detach()[0].permute(1, 2, 0).numpy()})
        imgList.append({'title' : 'Color pred', 'img' : c_pred})
        imgList.append({'title' : 'Color label', 'img' : c_label})
        displayImage(imgList, filename="att.png")
        break
    
    writer.flush()
    writer.close()

    print("Max mIOU: ", maxIOU)
