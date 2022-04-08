#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn, einsum
import torch.nn.functional as F


class conv_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_layer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x

    
class features(nn.Module):
    def __init__(self, in_ch):
        super(features, self).__init__()
        self.conv = nn.Sequential(
            conv_layer(in_ch, 64),
            conv_layer(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_layer(64, 128),
            conv_layer(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_layer(128, 256),
            conv_layer(256, 256),
            conv_layer(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_layer(256, 512),
            conv_layer(512, 512),
            conv_layer(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_layer(512, 512),
            conv_layer(512, 512),
            conv_layer(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
    def forward(self, x):
        out = self.conv(x)
        
        return out
    
    
class classifier(nn.Module):
    def __init__(self, ch_num, n_classes):
        super(classifier, self).__init__()
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch_num, ch_num),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(ch_num, n_classes),
        )
        

    def forward(self, z):
        pred = self.FC(z)
        
        return pred
