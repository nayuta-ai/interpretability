import torch
from torch import nn


class VGG16(nn.Module):

    def __init__(
            self, n_channels: int = 1, ch_num: int = 512, n_classes: int = 1):
        """
        Args:
            n_channels (int): number of input channel
            ch_num (int): number of hidden layer (Attention Layer) channel
            n_classes (int): number of output class
        """
        super(VGG16, self).__init__()
        # Feature extraction Layer
        self.features = features(n_channels, ch_num)

        # Classifier Layer
        self.classifier = classifier(ch_num, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """function to forward network
        Args:
            x (torch.Tensor): input data (image data 224*224)
        Returns:
            torch.Tensor: estimated classes from input data
        """
        y = self.features(x)
        pred = self.classifier(y)
        return pred


class conv_layer(nn.Module):
    """convolution layer"""
    def __init__(self, in_ch: int, out_ch: int):
        """
        Args:
            in_ch (int): number of input channel
            out_ch (int): number of output channel
        """
        super(conv_layer, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """function to forward network
        Args:
            x (torch.Tensor): input data
        Returns:
            torch.Tensor: convolutional data
        """
        x = self.conv_layer(x)
        return x


class features(nn.Module):
    """VGG16 feature extraction block"""
    def __init__(self, in_ch: int, ch_num: int):
        """
        Args:
            in_ch (int): number of input channel
            ch_num (int): number of hidden layer (Attention Layer) channel
        """
        super(features, self).__init__()
        self.conv = nn.Sequential(
            conv_layer(in_ch, ch_num//8),
            conv_layer(ch_num//8, ch_num//8),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_layer(ch_num//8, ch_num//4),
            conv_layer(ch_num//4, ch_num//4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_layer(ch_num//4, ch_num//2),
            conv_layer(ch_num//2, ch_num//2),
            conv_layer(ch_num//2, ch_num//2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_layer(ch_num//2, ch_num),
            conv_layer(ch_num, ch_num),
            conv_layer(ch_num, ch_num),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_layer(ch_num, ch_num),
            conv_layer(ch_num, ch_num),
            conv_layer(ch_num, ch_num),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """function to forward network
        Args:
            x (torch.Tensor): input data
        Returns:
            torch.Tensor: convolutional data
        """
        out = self.conv(x)
        return out


class classifier(nn.Module):
    """VGG16 classification block"""
    def __init__(self, ch_num: int, n_classes: int):
        """
        Args:
            ch_num (int): number of hidden layer (Attention Layer) channel
            n_classes (int): number of output class
        """
        super(classifier, self).__init__()
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch_num, ch_num),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(ch_num, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """function to forward network
        Args:
            x (torch.Tensor): input data
        Returns:
            torch.Tensor: convolutional data
        """
        pred = self.FC(z)
        return pred
