import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from adabelief_pytorch import AdaBelief
from torch_poly_lr_decay import PolynomialLRDecay


def my_loss(out, target):
    loss = (-out.log() * target).sum(dim=1).mean()
    return loss


# class CrossEntropy(nn.Module):
#     def __init__(self, ignore_label=-1, weight=None):
#         super(CrossEntropy, self).__init__()
#         self.ignore_label = ignore_label
#         self.criterion = nn.CrossEntropyLoss(
#             weight=weight,
#             ignore_index=ignore_label
#         )

#     def _forward(self, score, target):
#         ph, pw = score.size(2), score.size(3)
#         h, w = target.size(1), target.size(2)
#         if ph != h or pw != w:
#             score = F.interpolate(input=score, size=(
#                 h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

#         loss = self.criterion(score, target)

#         return loss

#     def forward(self, score, target):

#         if config.MODEL.NUM_OUTPUTS == 1:
#             score = [score]

#         weights = config.LOSS.BALANCE_WEIGHTS
#         assert len(weights) == len(score)

#         return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])
