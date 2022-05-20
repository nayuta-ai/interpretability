from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.get_dataset import ImageDataset
from data.transform import train_transform, val_transform


def get_dataloader(
        dataset: List[str], csv_file: pd.DataFrame, batch_size: int,
        type_dataset: str) -> DataLoader:
    """ A function to load data
    Args:
        dataset (List[str]): original dataset
        csv_file (pd.DataFrame): file contained the detail of original dataset
        batch_size (int): the size of batch
        type_dataset (str): the type of dataset such as train, val, and test
    """
    if type_dataset == "train":
        data = ImageDataset(
            dataset=dataset, csv_file=csv_file, transform=train_transform())
        return torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True, num_workers=4)
    elif type_dataset == "val":
        data = ImageDataset(
            dataset=dataset, csv_file=csv_file, transform=val_transform())
        return torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False, num_workers=4)
