# Libraries
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from adabelief_pytorch import AdaBelief
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import attention as att
import models
import train
from datasets.cityscapes import Cityscapes

# Global Variables
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')

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
parser.add_argument('--dataset_auto_labels_path', type=str, help='auto label path for extra images - cityscapes',
                    default="")
parser.add_argument('--dataset_infer_path', type=str, help='infer path', default="")
parser.add_argument('--dataset_save_infer_path', type=str, help='save infer path', default="")
parser.add_argument('--img_width', type=int, help='image width', required=True)
parser.add_argument('--img_height', type=int, help='image height', required=True)
parser.add_argument('--num_epochs', type=int, help='number of epochs of training', default=1)
parser.add_argument('--batch_size', type=int, help='batch size', default=1)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=0.01)
parser.add_argument('--GPU', type=str, help='GPU number', required=True)
parser.add_argument('--save_model_path', type=str, help='directory where to save model', default="")
parser.add_argument('--pre_train_model_path', type=str, help='directory to load pre trained model on Mapillary',
                    default="")
parser.add_argument('--load_model_path', type=str, help='directory where to load model from', default="")
parser.add_argument('--load_att_path', type=str, help='directory where to load attention model from', default="")
parser.add_argument('--metrics_path', type=str, help='directory where to save metrics from train and loss',
                    default="test")

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

MAX_IOU = -1
epochs = args.num_epochs


# =========== #
#  Functions  #
# =========== #
def displayImage(imgList, filename="test.png"):
    fig = plt.figure(figsize=(15, 15))

    nColumns = 3
    nLines = int(np.ceil(len(imgList) / 3.0))

    for index, img in enumerate(imgList):
        fig.add_subplot(nLines, nColumns, index + 1)
        plt.imshow(img['img'], interpolation='bilinear')
        plt.title(img['title'])

    # plt.show()
    plt.savefig(filename)


def iou_coef(pred, labels):
    smooth = 0.01
    intersection = np.sum(np.abs(labels[0:18] * pred[0:18]))
    # print("intersection: ", intersection)
    # intersection = np.sum(np.abs(labels*pred))
    union = np.sum(labels[0:18]) + np.sum(pred[0:18]) - intersection
    # print("union: ", union)
    iou = np.mean((intersection + smooth) / (union + smooth))
    return iou


def my_loss(out, target):
    # print("out shape:", out.shape)
    # print("target shape:", target.shape)
    loss = (-(out + 1e-5).log() * target)[:, 0:18].sum(dim=1).mean()
    return loss


# ====== #
#  Main  #
# ====== #
def main():
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # Datasets instances
    trainDataset = Cityscapes(args)
    valDataset = Cityscapes(args, eval=True)

    # trainDataset = attCityscapes(args)
    # valDataset = attCityscapes(args, eval=True)

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 4
    }

    val_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 4
    }

    # DataLoader declarations
    training_generator = torch.utils.data.DataLoader(trainDataset, **params)
    val_generator = torch.utils.data.DataLoader(valDataset, **val_params)

    # Network architecture
    model = models.wideResNet50()

    # Restore state dictionary
    if args.load_model_path != "":
        model.load_state_dict(torch.load(args.load_model_path))

    # model.load_state_dict(torch.load('test.pth'))
    # model.delete_features()
    # torch.save(model.state_dict(), "test.pth")

    # if torch.cuda.is_available():
    # model.cuda()

    # weight = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
    # weight_pth = torch.Tensor(weight).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weight_pth)
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(ignore_index=19)

    optimizer = AdaBelief(model.parameters(), lr=0.00001, eps=1e-16, betas=(0.9, 0.999))
    # lr_schedule = PolynomialLRDecay(optimizer, max_decay_steps=600000, end_learning_rate=0.00001, power=2.0)

    log_dir = None if (args.metrics_path == "") else args.metrics_path
    writer = SummaryWriter(log_dir=log_dir)

    if args.mode == "train_att":
        attModel = att.attModel((512, 1024))  # shape: resolution of imageH

        if args.load_att_path != "":
            attModel.load_state_dict(torch.load(args.load_att_path))

        train.trainAtt(args, model, attModel, training_generator, val_generator, device)
        return

    model.to(device)

    for e in tqdm(range(epochs)):
        model.train()
        for l_image, l_label in tqdm(training_generator, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            image, labels = l_image.to(device), l_label.to(device)

            optimizer.zero_grad()
            trunk_features, seg_pred = model(image)
            loss = my_loss(seg_pred, labels)

            # loss = criterion(out, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            # lr_schedule.step()
            # break

        # -----------validation------------
        model.eval()
        mIoU_sum = 0
        for l_image, l_label in tqdm(val_generator, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            image, labels = l_image.to(device), l_label.to(device)

            trunk_features, seg_pred = model(image)
            # print(out.shape)
            # print(out.cpu().detach().numpy().shape)
            mIoU = iou_coef(seg_pred.cpu().detach().numpy()[0], labels.cpu().detach().numpy()[0])
            mIoU_sum += mIoU
            # break

        mIoU_sum = float(mIoU_sum / len(val_generator))
        print("\n\n")
        print("mIoU_sum:", mIoU_sum)
        print("\n\n")

        writer.add_scalar('Loss/train', loss, e)
        writer.add_scalar('Accuracy/val', mIoU_sum, e)

        # Save model
        if mIoU_sum > maxIoU:
            maxIoU = mIoU_sum
            if e > 0.4 * epochs and args.save_model_path != "":
                torch.save(model.state_dict(), args.save_model_path)

    writer.flush()
    writer.close()

    print("Max mIoU: ", maxIoU)

    # -----------Final Evaluation------------
    model.eval()
    for l_image, l_label in val_generator:
        image, labels = l_image.to(device), l_label.to(device)
        imgList = [{'title': 'Original', 'img': l_image[0].permute(1, 2, 0)}]
        # print(l_image.shape)
        # print(l_label.shape)

        # print(out.shape)
        trunk_features, seg_pred = model(image)
        c_pred = seg_pred.cpu().detach().numpy()[0]  # .transpose((1, 2, 0))
        # print(c_pred.shape)

        c_pred = trainDataset.makeColorPred(c_pred)
        # c_label = trainDataset.makeColorPred(l_label[0])

        c_label = trainDataset.makeColorPred(l_label[0])
        # c_label = lb.cityscapes_pallete[l_label[0], :]

        imgList.append({'title': 'Color pred', 'img': c_pred})
        imgList.append({'title': 'Color label', 'img': c_label})
        # imgList.append({'title' : 'Pred', 'img' : pred_disp})
        displayImage(imgList)
        break


if __name__ == "__main__":
    main()
