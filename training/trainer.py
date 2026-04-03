"""Training and evaluation utilities for multi-task oral health model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from .dataset import get_dataloaders
from .loss import MultiTaskLoss
from .model import OralHealthModel

logger = logging.getLogger(__name__)


def _to_device(batch_images: torch.Tensor, batch_targets: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Move a batch and target dictionary to the target device."""
    images = batch_images.to(device, non_blocking=True)
    targets = {
        "mgi": batch_targets["mgi"].to(device, non_blocking=True).long(),
        "ohi": batch_targets["ohi"].to(device, non_blocking=True).long(),
        "gei": batch_targets["gei"].to(device, non_blocking=True).long(),
    }
    return images, targets


def train_one_epoch(
    model: OralHealthModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MultiTaskLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Multi-task model.
        loader: Training DataLoader.
        optimizer: Optimizer.
        loss_fn: MultiTaskLoss instance.
        device: Torch device.

    Returns:
        Dictionary with mean total and per-task losses.
    """
    model.train()

    total_loss_sum = 0.0
    mgi_sum = 0.0
    ohi_sum = 0.0
    gei_sum = 0.0
    n_batches = 0

    for batch_images, batch_targets in loader:
        images, targets = _to_device(batch_images, batch_targets, device)

        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        total_loss, task_losses = loss_fn(preds, targets)
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_sum += float(total_loss.item())
        mgi_sum += float(task_losses["mgi"].item())
        ohi_sum += float(task_losses["ohi"].item())
        gei_sum += float(task_losses["gei"].item())
        n_batches += 1

    if n_batches == 0:
        return {"total_loss": 0.0, "mgi_loss": 0.0, "ohi_loss": 0.0, "gei_loss": 0.0}

    return {
        "total_loss": total_loss_sum / n_batches,
        "mgi_loss": mgi_sum / n_batches,
        "ohi_loss": ohi_sum / n_batches,
        "gei_loss": gei_sum / n_batches,
    }


def evaluate(
    model: OralHealthModel,
    loader: torch.utils.data.DataLoader,
    loss_fn: MultiTaskLoss,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model performance on a validation set.

    Args:
        model: Multi-task model.
        loader: Validation DataLoader.
        loss_fn: MultiTaskLoss instance.
        device: Torch device.

    Returns:
        Dictionary containing average loss, per-task accuracy, per-task macro F1,
        and confusion matrices.
    """
    model.eval()

    total_loss_sum = 0.0
    n_batches = 0

    y_true = {"mgi": [], "ohi": [], "gei": []}
    y_pred = {"mgi": [], "ohi": [], "gei": []}

    with torch.no_grad():
        for batch_images, batch_targets in loader:
            images, targets = _to_device(batch_images, batch_targets, device)
            preds = model(images)
            total_loss, _ = loss_fn(preds, targets)

            total_loss_sum += float(total_loss.item())
            n_batches += 1

            for task in ("mgi", "ohi", "gei"):
                pred_labels = torch.argmax(preds[task], dim=1)
                y_pred[task].extend(pred_labels.cpu().numpy().tolist())
                y_true[task].extend(targets[task].cpu().numpy().tolist())

    avg_loss = total_loss_sum / max(1, n_batches)

    results: Dict[str, Any] = {
        "loss": avg_loss,
        "accuracy": {},
        "f1_macro": {},
        "confusion_matrices": {},
        "y_true": y_true,
        "y_pred": y_pred,
    }

    class_ranges = {
        "mgi": list(range(0, 5)),
        "ohi": list(range(0, 4)),
        "gei": list(range(0, 3)),
    }

    for task in ("mgi", "ohi", "gei"):
        task_true = np.array(y_true[task], dtype=np.int64)
        task_pred = np.array(y_pred[task], dtype=np.int64)

        if task_true.size == 0:
            results["accuracy"][task] = 0.0
            results["f1_macro"][task] = 0.0
            results["confusion_matrices"][task] = np.zeros((len(class_ranges[task]), len(class_ranges[task])), dtype=np.int64)
            continue

        results["accuracy"][task] = float(np.mean(task_true == task_pred))
        results["f1_macro"][task] = float(f1_score(task_true, task_pred, average="macro", zero_division=0))
        results["confusion_matrices"][task] = confusion_matrix(task_true, task_pred, labels=class_ranges[task])

    return results


@dataclass
class EarlyStopping:
    """Early stopping utility that saves the best model by validation loss."""

    patience: int
    checkpoint_path: Path

    def __post_init__(self) -> None:
        self.best_loss: float = float("inf")
        self.counter: int = 0

    def step(self, val_loss: float, model: OralHealthModel, optimizer: torch.optim.Optimizer, epoch: int) -> bool:
        """Update early stopping state and persist best checkpoint.

        Args:
            val_loss: Current validation loss.
            model: Model to checkpoint when improved.
            optimizer: Optimizer state.
            epoch: Epoch number.

        Returns:
            True if training should stop, otherwise False.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": val_loss,
                },
                self.checkpoint_path,
            )
            return False

        self.counter += 1
        return self.counter >= self.patience


def _head_parameters(model: OralHealthModel) -> List[torch.nn.Parameter]:
    """Collect trainable non-backbone model parameters."""
    params: List[torch.nn.Parameter] = []
    for module in (model.shared_projection, model.mgi_head, model.ohi_head, model.gei_head):
        params.extend(list(module.parameters()))
    return params


def _ensure_patient_column(patch_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure patch_df has patient_id column, deriving from filename if needed."""
    if "patient_id" in patch_df.columns:
        patch_df = patch_df.copy()
        patch_df["patient_id"] = patch_df["patient_id"].astype(str)
        return patch_df

    if "patch_path" not in patch_df.columns:
        raise ValueError("patch_df must contain patient_id or patch_path column.")

    out = patch_df.copy()
    out["patient_id"] = out["patch_path"].astype(str).map(lambda p: Path(p).stem.split("_")[0])
    return out


def train_kfold(
    patch_df: pd.DataFrame,
    config: Dict[str, Any],
    train_transform: Optional[Any],
    val_transform: Optional[Any],
) -> Dict[str, Any]:
    """Train model with patient-level Stratified K-fold cross-validation.

    Args:
        patch_df: Patch-level dataframe with labels and patient_id.
        config: Hyperparameter dictionary.
        train_transform: Albumentations train transform.
        val_transform: Albumentations val transform.

    Returns:
        Dictionary containing fold histories and aggregate metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_df = _ensure_patient_column(patch_df)

    patient_df = (
        patch_df.groupby("patient_id", as_index=False)
        .agg(mgi=("mgi", "first"), ohi=("ohi", "first"), gei=("gei", "first"))
    )

    skf = StratifiedKFold(
        n_splits=int(config["k_folds"]),
        shuffle=True,
        random_state=int(config.get("seed", 42)),
    )

    histories: List[List[Dict[str, Any]]] = []
    fold_summaries: List[Dict[str, Any]] = []

    checkpoint_dir = Path(str(config["checkpoint_dir"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_pat_idx, val_pat_idx) in enumerate(
        skf.split(patient_df["patient_id"], patient_df["mgi"]),
        start=1,
    ):
        logger.info("Starting fold %d/%d", fold_idx, int(config["k_folds"]))

        train_patients = set(patient_df.iloc[train_pat_idx]["patient_id"].astype(str).tolist())
        val_patients = set(patient_df.iloc[val_pat_idx]["patient_id"].astype(str).tolist())

        train_indices = patch_df.index[patch_df["patient_id"].isin(train_patients)].tolist()
        val_indices = patch_df.index[patch_df["patient_id"].isin(val_patients)].tolist()

        train_loader, val_loader = get_dataloaders(
            fold_train_indices=train_indices,
            fold_val_indices=val_indices,
            patch_df=patch_df,
            batch_size=int(config["batch_size"]),
            num_workers=int(config.get("num_workers", 0)),
            train_transform=train_transform,
            val_transform=val_transform,
        )

        model = OralHealthModel(config=config).to(device)
        loss_fn = MultiTaskLoss().to(device)

        model.freeze_backbone()
        head_params = _head_parameters(model) + list(loss_fn.parameters())
        optimizer = torch.optim.AdamW(head_params, lr=float(config["lr_heads"]))

        fold_checkpoint = checkpoint_dir / f"checkpoint_fold_{fold_idx}.pth"
        stopper = EarlyStopping(
            patience=int(config["early_stopping_patience"]),
            checkpoint_path=fold_checkpoint,
        )

        fold_history: List[Dict[str, Any]] = []
        num_epochs = int(config["num_epochs"])
        freeze_epochs = int(config["freeze_epochs"])

        for epoch in range(1, num_epochs + 1):
            if epoch == freeze_epochs + 1:
                model.unfreeze_last_n_blocks(4)
                backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
                head_params = _head_parameters(model)
                optimizer = torch.optim.AdamW(
                    [
                        {"params": backbone_params, "lr": float(config["lr_backbone"])},
                        {"params": head_params, "lr": float(config["lr_heads"])},
                        {"params": list(loss_fn.parameters()), "lr": float(config["lr_heads"])},
                    ]
                )

            train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_metrics = evaluate(model, val_loader, loss_fn, device)

            epoch_row = {
                "epoch": epoch,
                "train_total_loss": train_metrics["total_loss"],
                "train_mgi_loss": train_metrics["mgi_loss"],
                "train_ohi_loss": train_metrics["ohi_loss"],
                "train_gei_loss": train_metrics["gei_loss"],
                "val_loss": val_metrics["loss"],
                "val_acc_mgi": val_metrics["accuracy"]["mgi"],
                "val_acc_ohi": val_metrics["accuracy"]["ohi"],
                "val_acc_gei": val_metrics["accuracy"]["gei"],
                "val_f1_mgi": val_metrics["f1_macro"]["mgi"],
                "val_f1_ohi": val_metrics["f1_macro"]["ohi"],
                "val_f1_gei": val_metrics["f1_macro"]["gei"],
            }
            fold_history.append(epoch_row)

            logger.info(
                "Fold %d Epoch %d | train_loss=%.4f val_loss=%.4f f1(mgi/ohi/gei)=(%.4f, %.4f, %.4f)",
                fold_idx,
                epoch,
                train_metrics["total_loss"],
                val_metrics["loss"],
                val_metrics["f1_macro"]["mgi"],
                val_metrics["f1_macro"]["ohi"],
                val_metrics["f1_macro"]["gei"],
            )

            should_stop = stopper.step(
                val_loss=float(val_metrics["loss"]),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
            )
            if should_stop:
                logger.info("Early stopping triggered for fold %d at epoch %d", fold_idx, epoch)
                break

        histories.append(fold_history)

        # Load best checkpoint for summary metrics.
        best_ckpt = torch.load(fold_checkpoint, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        final_val_metrics = evaluate(model, val_loader, loss_fn, device)

        fold_summaries.append(
            {
                "fold": fold_idx,
                "best_val_loss": float(best_ckpt["best_val_loss"]),
                "final_val_accuracy": final_val_metrics["accuracy"],
                "final_val_f1": final_val_metrics["f1_macro"],
                "confusion_matrices": final_val_metrics["confusion_matrices"],
                "y_true": final_val_metrics["y_true"],
                "y_pred": final_val_metrics["y_pred"],
            }
        )

    avg_f1 = {
        task: float(np.mean([f["final_val_f1"][task] for f in fold_summaries]))
        for task in ("mgi", "ohi", "gei")
    }

    return {
        "histories": histories,
        "fold_summaries": fold_summaries,
        "avg_val_f1": avg_f1,
    }
