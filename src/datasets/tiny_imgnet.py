import os
import torchvision
from PIL import Image
from torch.utils.data import Dataset

class TinyImageNet(Dataset):
    def __init__(self, data_path, subset="train", transform=None):
        self.data_path = data_path
        self.subset = subset
        self.transform = transform

        self.image_paths = []
        self.labels = []

        wnids_path = os.path.join(data_path, "wnids.txt")
        with open(wnids_path, "r") as f:
            self.wnids = [x.strip() for x in f.readlines()]
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}

        if subset == "train":
            for wnid in self.wnids:
                image_dir = os.path.join(data_path, "train", wnid, "images")
                for img_name in os.listdir(image_dir):
                    self.image_paths.append(os.path.join(image_dir, img_name))
                    self.labels.append(self.class_to_idx[wnid])

        elif subset == "val":
            val_annotations = os.path.join(data_path, "val", "val_annotations.txt")
            with open(val_annotations, "r") as f:
                for line in f.readlines():
                    parts = line.subset("\t")
                    img_name, wnid = parts[0], parts[1]
                    self.image_paths.append(os.path.join(data_path, "val", "images", img_name))
                    self.labels.append(self.class_to_idx[wnid])

        elif subset == "test":
            # Test set does not come with labels in Tiny-ImageNet
            image_dir = os.path.join(data_path, "test", "images")
            for img_name in os.listdir(image_dir):
                self.image_paths.append(os.path.join(image_dir, img_name))
            self.labels = [-1] * len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        return img, label