import torch
from torch import nn


class VGG16(nn.Module):

    def __init__(
        self,
        n_channels: int = 1,
        ch_num: int = 512,
        n_classes: int = 1
    ) -> None:
        """
        Args:
            n_channels (int): number of input channel
            ch_num (int): number of hidden layer (Attention Layer) channel
            n_classes (int): number of output class
        """
        super(VGG16, self).__init__()
        # Feature extraction Layer
        self.features = Features(n_channels, ch_num)

        # Classifier Layer
        self.classifier = Classifier(ch_num, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """function to forward network
        Args:
            x (torch.Tensor): input data (image data 224*224)
        Returns:
            torch.Tensor: estimated classes from input data
        """
        y = self.features(x)
        return self.classifier(y)


class ConvLayer(nn.Module):
    """convolution layer"""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        """
        Args:
            in_ch (int): number of input channel
            out_ch (int): number of output channel
        """
        super(ConvLayer, self).__init__()
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
        return self.conv_layer(x)


class Features(nn.Module):
    """VGG16 feature extraction block"""
    def __init__(self, in_ch: int, hid_ch: int = None) -> None:
        """
        Args:
            in_ch (int): number of input channel
            hid_ch (int): number of hidden layer (Attention Layer) channel
        """
        super(Features, self).__init__()
        if not hid_ch:
            ch_num = [64, 128, 256, 512]
        else:
            ch_num = [hid_ch//8, hid_ch//4, hid_ch//2, hid_ch]
        self.layers = nn.Sequential(
            ConvLayer(in_ch, ch_num[0]),
            ConvLayer(ch_num[0], ch_num[0]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvLayer(ch_num[0], ch_num[1]),
            ConvLayer(ch_num[1], ch_num[1]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvLayer(ch_num[1], ch_num[2]),
            ConvLayer(ch_num[2], ch_num[2]),
            ConvLayer(ch_num[2], ch_num[2]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvLayer(ch_num[2], ch_num[3]),
            ConvLayer(ch_num[3], ch_num[3]),
            ConvLayer(ch_num[3], ch_num[3]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvLayer(ch_num[3], ch_num[3]),
            ConvLayer(ch_num[3], ch_num[3]),
            ConvLayer(ch_num[3], ch_num[3]),
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
        return self.layers(x)


class Classifier(nn.Module):
    """VGG16 classification block"""
    def __init__(self, ch_num: int, n_classes: int) -> None:
        """
        Args:
            ch_num (int): number of hidden layer (Attention Layer) channel
            n_classes (int): number of output class
        """
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
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
        return self.fc(z)
