import torch
import torch.nn as nn

from models.vgg16_gap import VGG16


def get_model(model_name: str) -> nn.Module:
    """Function to load model
    Args:
        model_name (str): name of model such as vgg16
    Returns:
        nn.Module: load model
    """
    if model_name == "vgg16":
        model = VGG16(n_channels=1, n_classes=1)
        model = torch.nn.DataParallel(model).cuda()
    else:
        raise ValueError(f"The {model_name} name doesn't exist.")
    return model
