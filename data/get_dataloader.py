from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import parse_yacs
from data.get_dataset import ImageDataset
from data.transform import train_transform, val_transform


def train_dataloader(files: List[str], csv_file: pd.DataFrame) -> DataLoader:
    """ A function for train dataloader
    Args:
        files(List[str]): original dataset
        csv_file(pd.DataFrame): file including the detail of original dataset
    Returns:
        DataLoader: train dataloader
    """
    args = parse_yacs()
    data = ImageDataset(files=files, csv_file=csv_file, transform=train_transform())
    return torch.utils.data.DataLoader(
       data,
       batch_size=args.TRAIN.BATCH_SIZE,
       shuffle=True,
       num_workers=4)

def val_dataloader(files: List[str], csv_file: pd.DataFrame) -> DataLoader:
    """ A function for val dataloader
    Args:
        files(List[str]): original dataset
        csv_file(pd.DataFrame): file including the detail of original dataset
    Returns:
        DataLoader: val dataloader
    """
    args = parse_yacs()
    data = ImageDataset(files=files, csv_file=csv_file, transform=val_transform())
    return torch.utils.data.DataLoader(
        data,
        batch_size=args.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=4)