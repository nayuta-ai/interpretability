import albumentations as album
import glob

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from config.const import CSV_PATH, DATA_PATH
from data.get_dataset import ImageDataset
from data.get_dataloader import train_dataloader, val_dataloader
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
    files = glob.glob(DATA_PATH)
    csv_file = pd.read_csv(CSV_PATH)
    files = np.array(files)[[0]].tolist()

    assert type(files) == list, \
    f"The type of files should be list, but {type(files)}"
    assert type(files[0]) == str, \
    f"The type of files[i] should be str, but {type(files[0])}"
    assert type(csv_file) == pd.DataFrame, \
    f"The type of data should be string, but {type(data[0][0])}"
    
    data = ImageDataset(files=files, csv_file=csv_file, transform=train_transform())
    
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
    assert len(data) == len(files), \
    f"The length of data should be {len(files)}, but {len(data)}"

def test_train_dataloader():
    files = glob.glob(DATA_PATH)
    csv_file = pd.read_csv(CSV_PATH)
    files = np.array(files)[[0]].tolist()
    
    assert type(files) == list, \
    f"The type of files should be list, but {type(files)}"
    assert type(files[0]) == str, \
    f"The type of files[i] should be str, but {type(files[0])}"
    assert type(csv_file) == pd.DataFrame, \
    f"The type of data should be string, but {type(data[0][0])}"
    
    dataloader = train_dataloader(files=files, csv_file=csv_file)
    
    assert type(dataloader) == DataLoader, \
    f"The type of data should be DataLoader, but {type(dataloader)}"

def test_val_dataloader():
    files = glob.glob(DATA_PATH)
    csv_file = pd.read_csv(CSV_PATH)
    files = np.array(files)[[0]].tolist()
    
    assert type(files) == list, \
    f"The type of files should be list, but {type(files)}"
    assert type(files[0]) == str, \
    f"The type of files[i] should be str, but {type(files[0])}"
    assert type(csv_file) == pd.DataFrame, \
    f"The type of data should be string, but {type(data[0][0])}"
    
    dataloader = val_dataloader(files=files, csv_file=csv_file)
    
    assert type(dataloader) == DataLoader, \
    f"The type of dataloader should be DataLoader, but {type(dataloader)}"
    assert len(dataloader) == len(files), \
    f"The length of dataloader should be {len(files)}, but {len(dataloader)}"
    for batch, samples in enumerate(dataloader):
        assert type(batch) == int, \
        f"The type of batch should be list, but {type(batch)}"
        assert type(samples) == list, \
        f"The type of samples should be list, but {type(samples)}"
        for i in range(len(samples)):
            if i == 0:
                assert type(samples[0][0]) == str, \
                f"The type of samples should be str, but {type(samples[0][0])}"
            elif i == 1:
                assert type(samples[1][0]) == torch.Tensor, \
                f"The type of samples should be Tensor, but {type(samples[1][0])}"
                assert len(samples[1][0][0]) == 224, \
                f"The row of samples should be 224, but {len(samples[1][0][0])}"
                assert len(samples[1][0][0][0]) == 224, \
                f"The col of samples should be 224, but {len(samples[1][0][0][0])}"
            else:
                assert type(samples[2][0]) == torch.Tensor, \
                f"The type of sample should be Tensor, but {type(samples[2][0])}"