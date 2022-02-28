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

import label as lb


class baseDataloader(torch.utils.data.Dataset):
    def __init__(self, args, eval=False):
        # TODO
        # 1. Initialize file paths or a list of file names.
        self.imagePath = args.dataset_images_path
        self.labelPath = args.dataset_labels_path
        self.pathExtraImages = args.dataset_extra_images_path
        self.pathAutoLabels = args.dataset_auto_labels_path
        self.img_width = args.img_width
        self.img_height = args.img_height
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.eval = eval

        if self.dataset == "cityscapes":
            addPathImg = "*/*_leftImg8bit.png"
            addPathLabel = "*/*_labelIds.png"
        elif self.dataset == "kitti":
            addPathImg = "image_2/*.png"
            addPathLabel = "semantic/*.png"

        if self.eval:  # validation
            self.images = sorted(glob(self.imagePath + "/val*/" + addPathImg))
            self.labels = sorted(glob(self.labelPath + "/val*/" + addPathLabel))
            if self.dataset == "kitti":
                self.images = sorted(glob(self.imagePath + "/train*/" + addPathImg))
                self.labels = sorted(glob(self.labelPath + "/train*/" + addPathLabel))

                self.images = self.images[int(0.95 * len(self.images)) :]
                self.labels = self.labels[int(0.95 * len(self.labels)) :]

        else:  # train
            self.images = sorted(glob(self.imagePath + "/train*/" + addPathImg))
            self.labels = sorted(glob(self.labelPath + "/train*/" + addPathLabel))

            if self.dataset == "cityscapes" and self.pathExtraImages != "" and self.pathAutoLabels != "":
                self.images += glob(self.pathExtraImages + "/train_extra/*/*.png")
                self.images = sorted(self.images)
                self.labels += glob(self.pathAutoLabels + "/*/*.png")
                self.labels = sorted(self.labels)

            elif self.dataset == "kitti":
                self.images = self.images[0 : int(0.95 * len(self.images))]
                self.labels = self.labels[0 : int(0.95 * len(self.labels))]

        ignore_label = 19

        self.label_mapping = {
            -1: ignore_label,
            0: ignore_label,
            1: ignore_label,
            2: ignore_label,
            3: ignore_label,
            4: ignore_label,
            5: ignore_label,
            6: ignore_label,
            7: 0,
            8: 1,
            9: ignore_label,
            10: ignore_label,
            11: 2,
            12: 3,
            13: 4,
            14: ignore_label,
            15: ignore_label,
            16: ignore_label,
            17: 5,
            18: ignore_label,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            29: ignore_label,
            30: ignore_label,
            31: 16,
            32: 17,
            33: 18,
        }

    # -------------- DATA AUG --------------- #
    def zoom(self, image, label):
        if image.shape[0] != 2 * self.img_height or image.shape[1] != 2 * self.img_width:
            image = cv2.resize(image, (2 * self.img_width, 2 * self.img_height))
            label = cv2.resize(label, (2 * self.img_height, 2 * self.img_height))

        # max_x = self.shape[1] - self.img_width
        # max_y = self.shape[0] - self.img_height

        x = np.random.randint(int(self.img_width / 4), int(self.img_width / 2))
        y = np.random.randint(int(self.img_height / 4), int(self.img_height / 2))

        image = image[y : y + self.img_height, x : x + self.img_width]
        label = label[y : y + self.img_height, x : x + self.img_width]

        return image, label

    @staticmethod
    def rotate(image, label):
        rot_angle = np.random.uniform(-0.0872665, 0.0872665)
        rsin = abs(math.sin(rot_angle))
        rcos = abs(math.cos(rot_angle))

        (h, w) = label.shape[:2]

        nh = int(h * rcos + w * rsin)
        nw = int(w * rcos + h * rsin)

        (cX, cY) = (nw // 2, nh // 2)

        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)

        M = cv2.getRotationMatrix2D((cX, cY), 180.0 * rot_angle / math.pi, 1.0)

        r_image = cv2.warpAffine(image, M, (nw, nh))
        r_label = cv2.warpAffine(label, M, (nw, nh))

        crop_h = int((nh - h) / 2)
        crop_w = int((nw - w) / 2)

        r_image = r_image[crop_h : h + crop_h, crop_w : w + crop_w]
        r_label = r_label[crop_h : h + crop_h, crop_w : w + crop_w]

        return r_image, r_label

    @staticmethod
    def color(image):
        gamma = np.random.uniform(0.9, 1.1)
        image = image ** gamma

        brightness = np.random.uniform(0.75, 1.25)
        image = image * brightness

        image = np.clip(image, 0, 255)

        return image

    def augmentData(self, image, label):
        if random() < 0.5 and self.dataset != "kitti":
            # image, label = self.zoom(image, label)
            new_size = (int(self.img_width / 2), int(self.img_height / 2))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            label = cv2.resize(label, new_size, interpolation=cv2.INTER_NEAREST)
        else:
            original_size = (self.img_width, self.img_height)
            image = cv2.resize(image, original_size, interpolation=cv2.INTER_AREA)
            label = cv2.resize(label, original_size, interpolation=cv2.INTER_NEAREST)

        # if(random() < 0.5):
        #     image, label = self.rotate(image, label)

        if random() < 0.5:
            image = self.color(image)

        return image, label

    def convertLabel(self, label):
        temp = label.copy()
        for id, trainID in self.label_mapping.items():
            label[temp == id] = trainID

        return label

    @staticmethod
    def normAndTranspImg(image):
        image = image.astype(np.float32)
        image = image / 255.0
        image = image.transpose((2, 0, 1))

        return image

    @staticmethod
    def makeColorPred(label):
        clabel = lb.cityscapes_pallete[np.argmax(label, axis=0), :]
        # clabel = clabel[:,:,0:3]

        return clabel

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.images)
