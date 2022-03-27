

# Whole pipeline:
# Download -> Extract -> Split -> Dataset -----> Transform -> dataloader
# 

from tkinter import Image
from typing import Optional, Sequence
from sklearn.utils import shuffle
import torch
import os, shutil
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
from pathlib import Path
from torch import nn

from torchvision import transforms

from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.finetuning import BaseFinetuning


class DownstreamDataloader(pl.LightningDataModule):
    def __init__(self, dataset_name: str, download: bool, task: str, batch_size: int, num_workers: int):
        super().__init__()
        self.dataset_name = dataset_name
        self.root_dir = Path('/img_data').joinpath(dataset_name) if 'per' in dataset_name else Path('/home/rick/offline_finetune').joinpath(dataset_name)
        self.download = download
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_urls = {
            'food101': 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz',
            'flowers102': '',
            'dtd': '',
            'cars': '',
            'cifar100': '',
            'pets': '',
            'sun397': '',
            'aircrafts': '',
            'caltech101': ''
        }
        self.dataset_transforms = {
            "linear_eval": {
                "train": self.linear_eval_train_transforms,
                "val": self.linear_eval_val_transforms,
                "test": self.linear_eval_val_transforms,
            },
            "finetune": {
                "train": self.finetune_train_transforms,
                "val": self.finetune_val_transforms,
                "test": self.finetune_val_transforms,
            }
        }


    def prepare_data(self):
        if self.download:
            download_and_extract_archive(self.dataset_urls[self.dataset_name], self.root_dir)

    def __dataloader(self, task: str, mode: str):
        if mode == 'val' or mode == 'test':
            dataset = self.create_dataset(self.data_path.joinpath(mode, 'val'), self.dataset_transforms[task][mode])
        else:
            dataset = self.create_dataset(self.data_path.joinpath(mode), self.dataset_transforms[task][mode])
        is_train = True if mode == 'train' else False
        return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=is_train)
    
    def create_dataset(self, root_path, transform):
        return ImageFolder(root_path, transform)

    def train_dataloader(self):
        return self.__dataloader(task=self.task, mode='train')

    def val_dataloader(self):
        return self.__dataloader(task=self.task, mode='val')

    def test_dataloader(self):
        return self.__dataloader(task=self.task, mode='test')

    @property
    def data_path(self):
        return Path(self.root_dir).joinpath("dataset")

    @property
    def linear_eval_train_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
            ]
        )

    @property
    def linear_eval_val_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
            ]
        )

    @property
    def finetune_train_transforms(self):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
            ]
        )
    
    @property
    def finetune_val_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(256, InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
            ]
        )
    
        
# --- Functions to prepare every individual dataset ---