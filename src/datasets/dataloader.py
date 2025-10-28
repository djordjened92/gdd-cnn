import numpy as np
import torch
import random
import torchvision
from typing import Tuple
from torchvision.datasets import DatasetFolder, FakeData
from torch.utils.data import DataLoader
from albumentations import Compose as AlbuCompose
from torchvision.transforms import Compose as TorchCompose
from torchvision.transforms import ToTensor

from datasets.aider import AIDER
from datasets.aiderv2 import AIDERV2
from datasets.cifar100 import CIFAR100
from datasets.tiny_imgnet import TinyImageNet

class CollateFnWrapper:
    def __init__(self,
                 target_size: tuple,
                 subset: str,
                 transforms: AlbuCompose,
                 norm_mean_std: Tuple,
                 device: torch.device):
        self.device = device
        self.transforms = transforms
        self.norm_mean_std = norm_mean_std
        self.subset = subset

        self.resize = torchvision.transforms.Resize(target_size)
        if self.norm_mean_std:
            self.normalize = torchvision.transforms.Normalize(self.norm_mean_std[0], self.norm_mean_std[1])
        self.to_tensor = ToTensor()

    def __call__(self, batch):
        images, labels = zip(*batch)
        if self.subset == 'train' and self.transforms is not None:
            if isinstance(self.transforms, AlbuCompose):
                images = np.stack([self.transforms(image=img.permute(1, 2, 0).detach().cpu().numpy())["image"] for img in images])
                images = torch.from_numpy(np.transpose(images, (0, 3, 1, 2))).to(torch.float32).to(self.device)
                images = images / 255.
            elif isinstance(self.transforms, TorchCompose):
                images = [self.transforms(img) for img in images]
                images = [self.resize(torchvision.transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))(self.to_tensor(img))) for img in images]
                images = torch.stack(images).to(self.device).to(torch.float32)
        else:
            images = [self.resize(img / 255. if isinstance(img, torch.Tensor) else self.to_tensor(img)) for img in images]
            images = torch.stack(images).to(self.device).to(torch.float32)
        labels = torch.tensor(list(labels), device=self.device)

        if self.norm_mean_std:
            images = self.normalize(images)

        return images, labels

def get_dataloader(dataset: DatasetFolder,
                   target_size: tuple,
                   batch_size: int,
                   shuffle: bool,
                   subset: str,
                   transforms: AlbuCompose | TorchCompose,
                   norm_mean_std: Tuple,
                   num_workers: int,
                   persistent_workers: bool,
                   pin_memory: bool,
                   device: torch.device):
    collate_fn = CollateFnWrapper(target_size,
                                  subset=subset,
                                  transforms=transforms,
                                  norm_mean_std=norm_mean_std,
                                  device=device)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            persistent_workers=persistent_workers,
                            pin_memory=pin_memory,
                            collate_fn=collate_fn)
    return dataloader
    
def get_dataset(dataset: str, 
                data_path: str, 
                target_size: tuple, 
                num_classes: int,
                subset:str, 
                seed: int,
                split: str='',
                k_folds: int=0,
                no_validation: bool=False,
                ) -> DatasetFolder:
    if dataset.upper() == "AIDER":
        return AIDER(data_path, target_size, subset, seed, split, k_folds, no_validation)
    elif dataset.upper() == "AIDERV2":
        return AIDERV2(data_path, target_size, subset)
    elif dataset.upper() == "CIFAR100":
        return CIFAR100(data_path, subset)
    elif dataset.upper() == "TINYIMAGENET":
        return TinyImageNet(data_path, subset)
    elif dataset.upper() == "FAKEDATA": # just for testing purposes
        return FakeData(size=100, image_size=(3, *target_size), num_classes=num_classes, transform=torchvision.transforms.ToTensor())
    else:
        raise ValueError(f"Dataset {dataset} not found.")