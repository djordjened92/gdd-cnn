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
                    transforms.RandomHorizontalFlip(),
                    # transforms.ColorJitter(
                    #     brightness=0.2,  # Jitter brightness by +/- 20%
                    #     contrast=(0.5, 1.5), # Contrast factor chosen uniformly from [0.5, 1.5]
                    #     saturation=0.3,  # Jitter saturation by +/- 30%
                    #     hue=0.1          # Jitter hue by +/- 0.1
                    # )
                ])
        return cifar_aug
    else:
        raise ValueError(f"Augmentation type {aug_type} not supported")