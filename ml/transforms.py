"""
Image transforms for dental photograph preprocessing.
Training transforms include augmentation; validation/inference transforms are deterministic.
"""

from torchvision import transforms


# Default image size (overridable via parameter)
IMAGE_SIZE = 300
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size=None):
    """Training transforms with data augmentation for dental images."""
    sz = image_size or IMAGE_SIZE
    return transforms.Compose([
        transforms.Resize((sz + 24, sz + 24)),
        transforms.RandomCrop(sz),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.05,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms(image_size=None):
    """Validation/inference transforms - deterministic."""
    sz = image_size or IMAGE_SIZE
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(image_size=None):
    """Inference transforms (same as validation)."""
    return get_val_transforms(image_size)
