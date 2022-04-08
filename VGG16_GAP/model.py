import torch

from .net_func import *

class model(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(model, self).__init__()

        # Feature extraction Layer
        self.features = features(n_channels)      
        
        # Attention Layer
        ch_num = 512
        
        # Classifier Layer
        self.classifier = classifier(ch_num, n_classes)

    def forward(self, x):
        y = self.features(x)
        pred = self.classifier(y)
        
        return pred