import os

import albumentations as album
import numpy as np
import pandas as pd
import cv2
import torch
from typing import List
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, files: List[str], csv_file: pd.DataFrame, transform: album.Compose) -> None:
        """ Initialization
        Args:
            files(List[str]): original dataset
            csv_file(pd.DataFrame): file including the detail of original dataset
            transform(album.Compose): the procedure of the data augmentation
        """
        self.files = files
        self.csv = csv_file
        self.transform = transform
        self.as_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self) -> int:
        """ Function for counting the number of data
        Returns:
            int: the number of files
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> (str, torch.Tensor, np.float32):
        """ Function for getting item
        Args:
            idx(int): the index of files
        Returns:
            str: the name of item
            torch.Tensor: the image data formatted Tensor
            np.float32: the median of data
        """
        img_file = self.files[idx]
        data = self.csv
        img = cv2.imread(img_file, 0)
        perf = np.nanmedian(data[data['name']==os.path.basename(img_file)[:10]]['Resist']).astype(np.float32)

        if 'T' in os.path.basename(img_file):
            img = ((img - 138.8) * 0.6 + 142.7).astype('uint8')
        
        augments = self.transform(image=img)
        img = self.as_tensor(augments['image'])
        return os.path.basename(img_file), img, perf