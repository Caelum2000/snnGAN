import torch
from torch import nn
from model_snn.nodes import *
from model_snn.encoders import *
import global_v as glv


class Discriminator_3_MP(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1),  # -4
            nn.BatchNorm2d(32),
            LIFNode()  # nn.LeakyReLU(0.2)
        )  # (24,24)
        self.pl1 = nn.AvgPool2d(2, stride=2)  # (12,12)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            LIFNode()  # nn.LeakyReLU(0.2)
        )
        self.pl2 = nn.AvgPool2d(2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),
            LIFNode()  # nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1),
            MPNode()  # nn.Sigmoid()
        )

    def forward(self, input, is_imgs=False):
        if is_imgs:
            # input are original images
            input = self.encoder(input)
            # print(input.shape)
        # input.shape = (n_steps,...)
        output = []
        for x in input:
            # print(x.shape)
            x = self.conv1(x)
            x = self.pl1(x)
            x = self.conv2(x)
            x = self.pl2(x)
            x = x.view(x.shape[0], -1)
            x = self.fc1(x)
            x = self.fc2(x)
            output.append(x)
        # output.shape = (n_steps, batch_size, 1)
        res_mem = output[-1] / glv.network_config['n_steps']  # (batch_size, 1)
        return res_mem
