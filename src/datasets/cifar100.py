import os
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

class CIFAR100(Dataset):
    def __init__(self, data_path: str, subset:str):
        self.dataset = datasets.CIFAR100(data_path, train=subset is 'train',
                                         download=True)
        self.data = self.dataset.data  # numpy array (N, H, W, C)
        self.targets = self.dataset.targets
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # convert numpy array to PIL Image
        return img, label