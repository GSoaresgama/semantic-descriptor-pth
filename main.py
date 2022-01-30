from cProfile import label
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

import dataloader as dl
import models
import label as lb
from torch.utils.tensorboard import SummaryWriter

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# ============== #
#  Args Parsing  #
# ============== #
def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description='SemSeg TensorFlow 2 implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--dataset', type=str, help='mapillary or cityscapes', required=True)
parser.add_argument('--dataset_images_path', type=str, help='image path', required=True)
parser.add_argument('--dataset_labels_path', type=str, help='label path', default="")
parser.add_argument('--dataset_extra_images_path', type=str, help='path for extra images - cityscapes', default="")
parser.add_argument('--dataset_auto_labels_path', type=str, help='auto label path for extra images - cityscapes', default="")
parser.add_argument('--dataset_infer_path', type=str, help='infer path', default="")
parser.add_argument('--dataset_save_infer_path', type=str, help='save infer path', default="")
parser.add_argument('--img_width', type=int, help='image width', required=True)
parser.add_argument('--img_height', type=int, help='image height', required=True)
parser.add_argument('--num_epochs', type=int, help='number of epochs of training', default=1)
parser.add_argument('--batch_size', type=int, help='batch size', default=1)
parser.add_argument('--learning_rate', type=float, help='inicial learning rate', default=0.01)
parser.add_argument('--GPU', type=str, help='GPU number', required=True)
parser.add_argument('--save_model_path', type=str, help='directory where to save model', default="")
parser.add_argument('--pre_train_model_path', type=str, help='directory to load pre trained model on Mapillary', default="")
parser.add_argument('--load_model_path', type=str, help='directory where to load model from', default="")
parser.add_argument('--load_att_path', type=str, help='directory where to load attention model from', default="")
parser.add_argument('--metrics_path', type=str, help='directory where to save metrics from train and loss', default="test")

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

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

# def iou_coef(y_true, y_pred, smooth=1):
#   intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
#   union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
#   iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
#   return iou

def iou_coef(pred, labels):
    smooth = 0.01
    intersection = np.sum(np.abs(labels[0:18]*pred[0:18]))
    # intersection = np.sum(np.abs(labels*pred))
    union = np.sum(labels) + np.sum(pred) - intersection
    iou = np.mean((intersection+smooth)/(union+smooth))
    return iou

def my_loss(out, target):
    # target[:][19] = np.zeros((target.shape[2], target.shape[3]))
    loss = ((-out+1e-5).log() * target)[:18].sum(dim=1).mean()
    return loss

def main():
    for arg in vars(args):
        print (arg, getattr(args, arg))

    trainDataset = dl.trainDataset(args)
    valDataset = dl.valDataset(args)

    params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 4}

    val_params = {'batch_size': 1,
        'shuffle': False,
        'num_workers': 4}

    training_generator = torch.utils.data.DataLoader(trainDataset, **params)
    val_generator = torch.utils.data.DataLoader(valDataset, **val_params)

    model = models.wideResnet50()
    model.to(device)
    # model.load_state_dict(torch.load('wr50v2.pth'))

    # weight = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
    # weight = torch.Tensor(weight)
    # criterion = nn.CrossEntropyLoss(weight=weight)
    # criterion = nn.CrossEntropyLoss(ignore_index=19)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = AdaBelief(model.parameters(), lr=0.01, eps=1e-16, betas=(0.9,0.999), weight_decouple = False, rectify = False)
    lr_schedule = PolynomialLRDecay(optimizer, max_decay_steps=300000, end_learning_rate=0.00001, power=2.0)

    # if torch.cuda.is_available():
    #     model.cuda()

    maxIOU = -1

    epochs = args.num_epochs
    writer = SummaryWriter()

    # print(model)

    i=0
    for e in tqdm(range(epochs)):

        for l_image, l_label in tqdm(training_generator, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            optimizer.zero_grad()

            image, labels = l_image.to(device), l_label.to(device)
            out = model(image)
            loss = my_loss(out, labels)
            # loss = criterion(out, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            lr_schedule.step()

            imgList = []
            imgList.append({'title' : 'Original', 'img' : l_image[0].permute(1, 2, 0)})
            # print(l_image.shape)
            # print(l_label.shape)

            # print(out.shape)
            out = model(image)
            c_pred = out.cpu().detach().numpy()[0]
            # print(c_pred.shape)
            # c_pred = np.transpose(c_pred,(1, 2, 0))
            # print(c_pred.shape)

            c_pred = trainDataset.makeColorPred(c_pred)
            c_label = trainDataset.makeColorPred(l_label[0])
            # c_label = trainDataset.makeColorPred(l_label[0])
            # print(l_label[0].cpu().detach().numpy())        
            # c_label = lb.cityscapes_pallete[l_label[0], :]
            
            imgList.append({'title' : 'Color pred', 'img' : c_pred})
            imgList.append({'title' : 'Color label', 'img' : c_label})
            # imgList.append({'title' : 'Pred', 'img' : pred_disp})
            displayImage(imgList, filename="train.png")
            break
        #-----------validation------------
        mIOUsum = 0
        for l_image, l_label in tqdm(val_generator, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            image, labels = l_image.to(device), l_label.to(device)
            out = model(image)
            mIOU = iou_coef(out.cpu().detach().numpy()[0], labels.cpu().detach().numpy()[0])
            mIOUsum += mIOU
            break
        mIOUsum = float(mIOUsum/len(val_generator))

        writer.add_scalar('Loss/train', loss, e)
        writer.add_scalar('Accuracy/val', mIOUsum, e)

        #save model
        if(mIOUsum > maxIOU):
            maxIOU = mIOUsum
            if(e>0.6*epochs and args.save_model_path != ""):
                torch.save(model.state_dict(), args.save_model_path)
                
    writer.flush()
    writer.close()

    print("Max mIOU: ", maxIOU)


    for l_image, l_label in training_generator:
        image, labels = l_image.to(device), l_label.to(device)
        imgList = []
        imgList.append({'title' : 'Original', 'img' : l_image[0].permute(1, 2, 0)})
        # print(l_image.shape)
        # print(l_label.shape)

        # print(out.shape)
        out = model(image)
        c_pred = out.cpu().detach().numpy()[0]
        # print(c_pred.shape)
        # c_pred = np.transpose(c_pred,(1, 2, 0))
        # print(c_pred.shape)

        c_pred = trainDataset.makeColorPred(c_pred)
        c_label = trainDataset.makeColorPred(l_label[0])
        # c_label = trainDataset.makeColorPred(l_label[0])
        # print(l_label[0].cpu().detach().numpy())        
        # c_label = lb.cityscapes_pallete[l_label[0], :]
        
        imgList.append({'title' : 'Color pred', 'img' : c_pred})
        imgList.append({'title' : 'Color label', 'img' : c_label})
        # imgList.append({'title' : 'Pred', 'img' : pred_disp})
        displayImage(imgList)
        break


if __name__ == "__main__":
    main()

