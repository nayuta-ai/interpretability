from __future__ import annotations

import torch

from models.vgg16_gap import VGG16


def get_model(network: str):
    """Function to load model
    Args:
        network (str): name of network such as vgg16
    Returns:
        model: type of VGG16 if network == "vgg16"
    """
    if network == "vgg16":
        model = VGG16(n_channels=1, n_classes=1)
        model = torch.nn.DataParallel(model).cuda()
    return model
