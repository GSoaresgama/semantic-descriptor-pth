# Libraries
from cProfile import label
from nis import match
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
import train
from datasets.cityscapes import Cityscapes, attCityscapes

# Global Variables
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# ============== #
#  Args Parsing  #
# ============== #
def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description="SemSeg TensorFlow 2 implementation.", fromfile_prefix_chars="@")
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument("--mode", type=str, help="train or test", default="train")
parser.add_argument("--ft_flag", type=bool, help="train or test", default=False)
parser.add_argument("--dataset", type=str, help="mapillary or cityscapes", required=True)
parser.add_argument("--dataset_images_path", type=str, help="image path", required=True)
parser.add_argument("--dataset_labels_path", type=str, help="label path", default="")
parser.add_argument("--dataset_extra_images_path", type=str, help="path for extra images - cityscapes", default="")
parser.add_argument("--dataset_auto_labels_path", type=str, help="auto label path for extra images - cityscapes", default="")
parser.add_argument("--dataset_infer_path", type=str, help="infer path", default="")
parser.add_argument("--dataset_save_infer_path", type=str, help="save infer path", default="")
parser.add_argument("--img_width", type=int, help="image width", required=True)
parser.add_argument("--img_height", type=int, help="image height", required=True)
parser.add_argument("--num_epochs", type=int, help="number of epochs of training", default=1)
parser.add_argument("--batch_size", type=int, help="batch size", default=1)
parser.add_argument("--learning_rate", type=float, help="inicial learning rate", default=0.01)
parser.add_argument("--GPU", type=str, help="GPU number", required=True)
parser.add_argument("--save_model_path", type=str, help="directory where to save model", default="")
parser.add_argument("--pre_train_model_path", type=str, help="directory to load pre trained model on Mapillary", default="")
parser.add_argument("--load_model_path", type=str, help="directory where to load model from", default="")
parser.add_argument("--load_att_path", type=str, help="directory where to load attention model from", default="")
parser.add_argument("--metrics_path", type=str, help="directory where to save metrics from train and loss", default="test")

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = "@" + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()


# ====== #
#  Main  #
# ====== #
def main():
    for arg in vars(args):
        print(arg, getattr(args, arg))

    params = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 4}
    val_params = {"batch_size": 1, "shuffle": False, "num_workers": 4}

    model = models.wideResnet50()

    if args.load_model_path != "":
        model.load_state_dict(torch.load(args.load_model_path))

    if args.mode == "train":
        trainDataset = Cityscapes(args)
        valDataset = Cityscapes(args, eval=True)
        training_generator = torch.utils.data.DataLoader(trainDataset, **params)
        val_generator = torch.utils.data.DataLoader(valDataset, **val_params)

        print(len(training_generator))
        print(len(val_generator))

        train.trainTrunk(args, model, training_generator, val_generator, device)

    elif args.mode == "train_att":
        trainDataset = attCityscapes(args)
        valDataset = attCityscapes(args, eval=True)
        training_generator = torch.utils.data.DataLoader(trainDataset, **params)
        val_generator = torch.utils.data.DataLoader(valDataset, **val_params)

        attModel = att.attModel ((args.img_height, args.img_width))

        if args.load_att_path != "":
            attModel.load_state_dict(torch.load(args.load_att_path))

        print(len(training_generator))
        print(len(val_generator))

        train.trainAtt(args, model, attModel, training_generator, val_generator, device)


if __name__ == "__main__":
    main()
