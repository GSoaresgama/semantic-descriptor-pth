from cProfile import label
import math
from random import random
from sympy import arg
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as fn

import os
from glob import glob
import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F
from datasets.dataloader import baseDataloader

import label as lb


class cityscapes(baseDataloader):
    # TODO: Leitura e dataAugmentation está todo implementado em OpenCV, usar transforms
    def __getitem__(self, index):

        imgPath = self.images[index]
        labelPath = self.labels[index]

        image = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        label = cv2.imread(labelPath, cv2.IMREAD_GRAYSCALE)

        self.shape = image.shape
        image = image[:, :, ::-1]
        if not self.eval:
            image, label = self.augmentData(image, label)
        else:
            image = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
            label = cv2.resize(label, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

        label = self.convertLabel(label)

        # normalizer image
        image = image.astype(np.float32)
        image = image / 255.0
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        label = label.long()

        label = F.one_hot(label, num_classes=20)
        label = label.permute((2, 0, 1))
        # print("label one hot: ", label.shape)
        # label = label.to(torch.float32)
        # label[19,:,:] = 0
        return image, label


class attCityscapes(baseDataloader):
    def __getitem__(self, index):

        imgPath = self.images[index]
        labelPath = self.labels[index]
        # print(imgPath)
        # print(labelPath)
        image = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        label = cv2.imread(labelPath, cv2.IMREAD_GRAYSCALE)

        self.shape = image.shape
        image = image[:, :, ::-1]

        if not self.eval and random() < 0.5:
            image = self.color(image)

        if image.shape[0] != self.img_height or image.shape[1] != self.img_width:
            imageH = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
            label = cv2.resize(label, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        else:
            imageH = image.copy()

        imageL = cv2.resize(image, (int(self.img_width / 2), int(self.img_height / 2)), interpolation=cv2.INTER_AREA)

        label = self.convertLabel(label)

        # normalize images
        imageL = self.normAndTranspImg(imageL)
        imageH = self.normAndTranspImg(imageH)

        imageL = torch.from_numpy(imageL)
        imageH = torch.from_numpy(imageH)

        label = torch.from_numpy(label)
        label = label.long()

        label = F.one_hot(label, num_classes=20)
        label = label.permute((2, 0, 1))

        return imageL, imageH, label
