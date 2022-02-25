from audioop import bias
from cProfile import label
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


class attModel(nn.Module):
    def __init__(self, shape, pretrained=True):
        super().__init__()

        self.shape = shape
        # self.features = nn.ModuleList(self.wr50v2.children())[:-1]
        # print(trunk)

        # TODO: Porque você pulou de 2048 para 512? Na trunk, você saiu de 2048 para 1024.
        self.attHead = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )

        for m in self.attHead:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=1)

    def forward(self, input_trunk, predL, predH):
        resize = torchvision.transforms.Resize(self.shape)

        attMask = self.attHead(input_trunk)

        attL = torch.mul(predL, attMask)
        attH = torch.mul((1 - resize(attMask)), predH)

        output = torch.add(resize(attL), attH)

        return attMask.detach(), output
