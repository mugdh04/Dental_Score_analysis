"""
Image transforms for dental photograph preprocessing.
Uses DINOv2-compatible preprocessing (224px, ImageNet normalization).
Training transforms include augmentation; validation/inference transforms are deterministic.
"""

from torchvision import transforms

# DINOv2 uses 518px input by default, but 224px works well and is faster on CPU
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size=None):
    """Training transforms with augmentation for DINOv2 feature extraction."""
    sz = image_size or IMAGE_SIZE
    return transforms.Compose([
        transforms.RandomResizedCrop(sz, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])


def get_val_transforms(image_size=None):
    """Validation/inference transforms - deterministic."""
    sz = image_size or IMAGE_SIZE
    return transforms.Compose([
        transforms.Resize(int(sz * 1.05)),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(image_size=None):
    """Inference transforms (same as validation)."""
    return get_val_transforms(image_size)


def get_tta_transforms(image_size=None):
    """Test-Time Augmentation transforms — returns list of transform pipelines."""
    sz = image_size or IMAGE_SIZE
    base = [
        transforms.Resize(int(sz * 1.05)),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return [
        transforms.Compose(base),  # original
        transforms.Compose([
            transforms.Resize(int(sz * 1.05)),
            transforms.CenterCrop(sz),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    ]
