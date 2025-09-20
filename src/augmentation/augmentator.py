from augmentation.aider import create_aider_augmentations
from torchvision import transforms

def select_augmentation(aug_type: str, target_size: tuple, p: float):
    if aug_type is None:
        return None
    if aug_type.upper() == 'AIDER':
        return create_aider_augmentations(*target_size, p)
    elif aug_type.upper() == 'CIFAR':
        cifar_aug = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip()
                ])
        return cifar_aug
    elif aug_type.upper() == 'TinyImageNet':
        cifar_aug = transforms.Compose([
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip()
                ])
        return cifar_aug
    else:
        raise ValueError(f"Augmentation type {aug_type} not supported")