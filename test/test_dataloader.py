import glob

import albumentations as album
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from config.const import CSV_PATH, DATA_PATH
from data.get_dataloader import get_dataloader
from data.get_dataset import ImageDataset
from data.transform import train_transform, val_transform


def test_train_transform():
    transform = train_transform()

    assert type(transform) == album.Compose, \
        f"The type of transform should be DataLoader, but {type(transform)}"


def test_val_transform():
    transform = val_transform()

    assert type(transform) == album.Compose, \
        f"The type of transform should be DataLoader, but {type(transform)}"


def test_ImageDataset():
    dataset = glob.glob(DATA_PATH)
    csv_file = pd.read_csv(CSV_PATH)
    dataset = np.array(dataset)[[0]].tolist()

    assert type(dataset) == list, \
        f"The type of data should be list, but {type(dataset)}"
    assert type(dataset[0]) == str, \
        f"The type of data[i] should be str, but {type(dataset[0])}"
    assert type(csv_file) == pd.DataFrame, \
        f"The type of csv_file should be string, but {type(csv_file)}"

    # data: tuple(str, torch.Tensor, np.float32)
    data = ImageDataset(
        dataset=dataset, csv_file=csv_file, transform=train_transform())

    assert type(data) == ImageDataset, \
        f"The type of data should be ImageDataset, but {type(data)}"
    assert type(data[0]) == tuple, \
        f"The type of data should be tuple, but {type(data[0])}"
    assert type(data[0][0]) == str, \
        f"The type of data should be string, but {type(data[0][0])}"
    assert type(data[0][1]) == torch.Tensor, \
        f"The type of data should be Tensor, but {type(data[0][1])}"
    assert type(data[0][2]) == np.float32, \
        f"The type of data should be float, but {type(data[0][2])}"
    assert len(data) == len(dataset), \
        f"The length of data should be {len(data)}, but {len(dataset)}"


@pytest.mark.parametrize(
    "batch_size, type_dataset", [
        (32, "train"),
        (32, "val"),
    ]
)
def test_get_dataloader(batch_size: int, type_dataset: str):
    """ A function to test the function called "get_dataloader"
    Args:
        batch_size (int): the size of batch
        type_dataset (str): the type of dataset such as train, val, and test
    """
    dataset = glob.glob(DATA_PATH)
    csv_file = pd.read_csv(CSV_PATH)
    dataset = np.array(dataset)[[0]].tolist()

    assert type(dataset) == list, \
        f"The type of dataset should be list, but {type(dataset)}"
    assert type(dataset[0]) == str, \
        f"The type of dataset[i] should be str, but {type(dataset[0])}"
    assert type(csv_file) == pd.DataFrame, \
        f"The type of csv_file should be string, but {type(csv_file[0][0])}"

    dataloader = get_dataloader(
        dataset=dataset, csv_file=csv_file, batch_size=batch_size,
        type_dataset=type_dataset)

    assert type(dataloader) == DataLoader, \
        f"The type of dataloader should be DataLoader, but {type(dataloader)}"
    assert len(dataloader) == len(dataset), \
        f"The length of dataloader should be \
        {len(dataset)}, but {len(dataloader)}"
    for batch, samples in enumerate(dataloader):
        assert type(batch) == int, \
            f"The type of batch should be list, but {type(batch)}"
        assert type(samples) == list, \
            f"The type of samples should be list, but {type(samples)}"
        for i in range(len(samples)):
            if i == 0:
                assert type(samples[0][0]) == str, \
                    f"The type of samples should be str, \
                    but {type(samples[0][0])}"
            elif i == 1:
                assert type(samples[1][0]) == torch.Tensor, \
                    f"The type of samples should be Tensor, \
                    but {type(samples[1][0])}"
                assert len(samples[1][0]) == 1, \
                    f"The size of samples should be 1, \
                    but {len(samples[1][0])}"
                assert len(samples[1][0][0]) == 224, \
                    f"The row of samples should be 224, \
                    but {len(samples[1][0][0])}"
                assert len(samples[1][0][0][0]) == 224, \
                    f"The col of samples should be 224, \
                    but {len(samples[1][0][0][0])}"
            else:
                assert type(samples[2][0]) == torch.Tensor, \
                    f"The type of sample should be Tensor, \
                    but {type(samples[2][0])}"
                assert len(samples[2]) == 1, \
                    f"The length of sample should be 1, \
                    but {len(samples[2])}"
