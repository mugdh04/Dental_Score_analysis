"""Training package for data loading, model definition, losses, and training loops."""

from .dataset import (
    load_dataset,
    MultiViewPatientDataset,
    PatchDataset,
    get_dataloaders,
    build_weighted_sampler,
)
from .model import OralHealthModel
from .loss import MultiTaskLoss, FocalLoss, OrdinalLoss
from .trainer import train_one_epoch, evaluate, EarlyStopping, train_kfold, compute_class_weights

__all__ = [
    # Dataset
    "load_dataset",
    "MultiViewPatientDataset",
    "PatchDataset",
    "get_dataloaders",
    "build_weighted_sampler",
    # Model
    "OralHealthModel",
    # Loss
    "MultiTaskLoss",
    "FocalLoss",
    "OrdinalLoss",
    # Training
    "train_one_epoch",
    "evaluate",
    "EarlyStopping",
    "train_kfold",
    "compute_class_weights",
]
