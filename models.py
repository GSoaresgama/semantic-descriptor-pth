from cProfile import label
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


class wideResnet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        wr50v2 = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
        for param in wr50v2.parameters():
            param.requires_grad = True
        # print(wr50v2)
        # here we get all the modules(layers) before the fc layer at the end
        # note that currently at pytorch 1.0 the named_children() is not supported
        # and using that instead of children() will fail with an error
        self.new_model = nn.Sequential(*list(wr50v2.children())[:-2])
        # print(self.new_model)
        # self.features = nn.ModuleList(wr50v2.children())[:-2]
        # in_features = wr50v2.fc.in_features
        self.upconvs = nn.Sequential(
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
            nn.Conv2d(64, 20, kernel_size=(1, 1), stride=(1, 1)),
            nn.Softmax(dim=1)
        )

        for m in self.upconvs:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=1)

    def delete_features(self):
        del self.features

    def forward(self, input_imgs):
        trunk = self.new_model(input_imgs)
        output = self.upconvs(trunk)

        return trunk, output
