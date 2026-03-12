"""
Image transforms for dental photograph preprocessing.
Uses DINOv2-compatible preprocessing (ImageNet normalization).
Training transforms include heavy augmentation to combat small dataset (201 samples).
Supports both 224px (fast) and 518px (DINOv2 native, best quality).
"""

from torchvision import transforms

# DINOv2 native resolution is 518px; 224px is faster for quick experiments
IMAGE_SIZE = 518
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size=None):
    """Heavy training augmentation for fine-tuning on tiny dataset."""
    sz = image_size or IMAGE_SIZE
    return transforms.Compose([
        transforms.RandomResizedCrop(sz, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.06),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])


def get_val_transforms(image_size=None):
    """Validation/inference transforms - deterministic."""
    sz = image_size or IMAGE_SIZE
    return transforms.Compose([
        transforms.Resize(int(sz * 1.04), interpolation=transforms.InterpolationMode.BICUBIC),
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
        transforms.Resize(int(sz * 1.04), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return [
        transforms.Compose(base),  # original
        transforms.Compose([
            transforms.Resize(int(sz * 1.04), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(sz),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    ]
