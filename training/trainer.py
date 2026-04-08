"""Training and evaluation engine for multi-task oral health model.

Changes from original:
  - Multi-view triplet batches instead of patch-based batches
  - Macro-F1 (not val-loss) used for early stopping and best-model selection
  - Class-weighted + focal + ordinal loss from training/loss.py
  - Saves ensemble_config.json to models/ folder after all folds complete
  - Patient-level stratified K-fold (prevents data leakage)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

from .dataset import MultiViewPatientDataset, get_dataloaders, load_dataset
from .loss import MultiTaskLoss
from .model import OralHealthModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _triplet_to_device(
    batch: Tuple,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Move a multi-view batch to the target device.

    Args:
        batch: Tuple of (frontal, left, right, labels_dict).
        device: Target device.

    Returns:
        (frontal, left, right, targets) all on device.
    """
    frontal, left, right, labels = batch
    targets = {
        "mgi": labels["mgi"].to(device, non_blocking=True).long(),
        "ohi": labels["ohi"].to(device, non_blocking=True).long(),
        "gei": labels["gei"].to(device, non_blocking=True).long(),
    }
    return (
        frontal.to(device, non_blocking=True),
        left.to(device, non_blocking=True),
        right.to(device, non_blocking=True),
        targets,
    )


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: OralHealthModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MultiTaskLoss,
    device: torch.device,
    scaler: Optional[amp.GradScaler] = None,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train model for one epoch with optional mixed-precision.

    Args:
        model: Multi-view dental model.
        loader: Training DataLoader yielding triplets.
        optimizer: Optimizer.
        loss_fn: MultiTaskLoss.
        device: Training device.
        scaler: GradScaler for AMP (None = disabled).
        grad_clip: Max gradient norm for clipping.

    Returns:
        Dict with mean losses: total_loss, mgi_loss, ohi_loss, gei_loss.
    """
    model.train()

    totals = {"total": 0.0, "mgi": 0.0, "ohi": 0.0, "gei": 0.0}
    n_batches = 0

    for batch in loader:
        frontal, left, right, targets = _triplet_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            preds = model(frontal, left, right)
            total_loss, task_losses = loss_fn(preds, targets)

        if scaler and use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        totals["total"] += float(total_loss.item())
        totals["mgi"] += float(task_losses["mgi"].item())
        totals["ohi"] += float(task_losses["ohi"].item())
        totals["gei"] += float(task_losses["gei"].item())
        n_batches += 1

    if n_batches == 0:
        return {"total_loss": 0.0, "mgi_loss": 0.0, "ohi_loss": 0.0, "gei_loss": 0.0}

    return {
        "total_loss": totals["total"] / n_batches,
        "mgi_loss":   totals["mgi"]   / n_batches,
        "ohi_loss":   totals["ohi"]   / n_batches,
        "gei_loss":   totals["gei"]   / n_batches,
    }


# ---------------------------------------------------------------------------
# Evaluation with optional TTA
# ---------------------------------------------------------------------------

def evaluate(
    model: OralHealthModel,
    loader: torch.utils.data.DataLoader,
    loss_fn: MultiTaskLoss,
    device: torch.device,
    tta_steps: int = 1,
) -> Dict[str, Any]:
    """Evaluate model on validation set.

    Args:
        model: Multi-view model.
        loader: Validation DataLoader.
        loss_fn: MultiTaskLoss.
        device: Evaluation device.
        tta_steps: If > 1, run each batch multiple times and average
                   softmax outputs (Test-Time Augmentation).

    Returns:
        Dict with loss, accuracy/f1_macro/confusion matrices per task,
        and raw y_true/y_pred arrays.
    """
    model.eval()

    total_loss_sum = 0.0
    n_batches = 0
    y_true: Dict[str, List[int]] = {"mgi": [], "ohi": [], "gei": []}
    y_pred: Dict[str, List[int]] = {"mgi": [], "ohi": [], "gei": []}

    with torch.no_grad():
        for batch in loader:
            frontal, left, right, targets = _triplet_to_device(batch, device)

            if tta_steps > 1:
                # Sum up softmax probabilities across TTA passes, take argmax at the end
                task_probs = {
                    "mgi": torch.zeros(frontal.size(0), loss_fn.ce_mgi.weight.shape[0]
                        if hasattr(loss_fn.ce_mgi, 'weight') and loss_fn.ce_mgi.weight is not None
                        else 5).to(device),
                    "ohi": torch.zeros(frontal.size(0), loss_fn.ce_ohi.weight.shape[0]
                        if hasattr(loss_fn.ce_ohi, 'weight') and loss_fn.ce_ohi.weight is not None
                        else 4).to(device),
                    "gei": torch.zeros(frontal.size(0), loss_fn.ce_gei.weight.shape[0]
                        if hasattr(loss_fn.ce_gei, 'weight') and loss_fn.ce_gei.weight is not None
                        else 4).to(device),
                }
                for _ in range(tta_steps):
                    preds = model(frontal, left, right)
                    for k in ("mgi", "ohi", "gei"):
                        probs = torch.softmax(preds[k], dim=1)
                        if task_probs[k].shape[1] != probs.shape[1]:
                            task_probs[k] = torch.zeros_like(probs)
                        task_probs[k] += probs / tta_steps

                # Compute loss on first pass logits (for tracking)
                preds_for_loss = model(frontal, left, right)
                total_loss, _ = loss_fn(preds_for_loss, targets)
                total_loss_sum += float(total_loss.item())
                n_batches += 1

                for task in ("mgi", "ohi", "gei"):
                    pred_labels = task_probs[task].argmax(dim=1)
                    y_pred[task].extend(pred_labels.cpu().numpy().tolist())
                    y_true[task].extend(targets[task].cpu().numpy().tolist())

            else:
                preds = model(frontal, left, right)
                total_loss, _ = loss_fn(preds, targets)
                total_loss_sum += float(total_loss.item())
                n_batches += 1

                for task in ("mgi", "ohi", "gei"):
                    pred_labels = preds[task].argmax(dim=1)
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

    class_ranges = {"mgi": list(range(5)), "ohi": list(range(4)), "gei": list(range(4))}
    for task in ("mgi", "ohi", "gei"):
        t = np.array(y_true[task], dtype=np.int64)
        p = np.array(y_pred[task], dtype=np.int64)
        if t.size == 0:
            results["accuracy"][task] = 0.0
            results["f1_macro"][task] = 0.0
            results["confusion_matrices"][task] = np.zeros(
                (len(class_ranges[task]), len(class_ranges[task])), dtype=np.int64
            )
            continue
        results["accuracy"][task] = float(np.mean(t == p))
        results["f1_macro"][task] = float(f1_score(t, p, average="macro", zero_division=0))
        results["confusion_matrices"][task] = confusion_matrix(t, p, labels=class_ranges[task])

    return results


# ---------------------------------------------------------------------------
# Early stopping — monitors macro-F1 (maximise)
# ---------------------------------------------------------------------------

@dataclass
class EarlyStopping:
    """Early stopping that saves the best model by *validation macro-F1*.

    We monitor F1 rather than loss because with imbalanced data, loss can keep
    decreasing while the model continues collapsing onto majority classes.

    Args:
        patience: Number of epochs with no F1 improvement before stopping.
        checkpoint_path: Where to save the best checkpoint.
    """

    patience: int
    checkpoint_path: Path
    best_f1: float = field(default=0.0, init=False)
    counter: int = field(default=0, init=False)

    def step(
        self,
        avg_f1: float,
        model: OralHealthModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: MultiTaskLoss,
        epoch: int,
        config: Dict[str, Any],
    ) -> bool:
        """Update state and save checkpoint if F1 improved.

        Args:
            avg_f1: Average macro-F1 across all tasks.
            model: Model to checkpoint.
            optimizer: Optimizer state.
            loss_fn: Loss function state (contains learnable log_vars).
            epoch: Current epoch number.
            config: Configuration dict (saved in checkpoint).

        Returns:
            True if training should stop.
        """
        if avg_f1 > self.best_f1:
            self.best_f1 = avg_f1
            self.counter = 0
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_fn_state_dict": loss_fn.state_dict(),
                    "best_val_f1": avg_f1,
                    "config": config,
                },
                self.checkpoint_path,
            )
            logger.info("Checkpoint saved (epoch=%d, val_f1=%.4f) → %s", epoch, avg_f1, self.checkpoint_path)
            return False

        self.counter += 1
        if self.counter >= self.patience:
            return True
        return False


# ---------------------------------------------------------------------------
# Computes class weights from an array of labels
# ---------------------------------------------------------------------------

def compute_class_weights(
    labels: Sequence[int],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute inverse-frequency class weights.

    Args:
        labels: Flat list of integer class labels.
        num_classes: Total number of classes (may be more than unique in labels).
        device: Target device for the returned tensor.

    Returns:
        Float tensor of shape (num_classes,).
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    # Avoid division by zero for unseen classes (give them a large weight)
    counts = np.where(counts == 0, 0.5, counts)
    weights = counts.sum() / (num_classes * counts)
    # Cap maximum weight to avoid extreme values for very rare classes
    weights = np.clip(weights, a_min=None, a_max=10.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# K-Fold training orchestrator
# ---------------------------------------------------------------------------

def train_kfold(
    records: List[Dict[str, Any]],
    config: Dict[str, Any],
    train_transform,
    val_transform,
) -> Dict[str, Any]:
    """Train model with patient-level Stratified K-fold cross-validation.

    Strategy:
        Phase 1 (epochs 1..freeze_epochs): Backbone frozen, only heads trained.
        Phase 2 (remaining epochs):        Unfreeze last N backbone blocks at
                                           backbone_lr (10–20x lower than heads).

    Uses WeightedRandomSampler + focal+ordinal+class-weighted loss to combat
    the class-imbalance / predict-only-extremes failure mode.

    Args:
        records: Output of load_dataset() — list of image-level dicts.
        config: Hyperparameter configuration dict.
        train_transform: Albumentations transform for training split.
        val_transform: Albumentations transform for validation split.

    Returns:
        Dict with:
          histories      — per-fold per-epoch metrics
          fold_summaries — per-fold best results
          avg_val_f1     — averaged macro-F1 across folds per task
          checkpoint_paths — list of saved checkpoint file paths
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    checkpoint_dir = Path(str(config["checkpoint_dir"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Build the full dataset (both train and val transforms)
    # -----------------------------------------------------------------------
    train_ds = MultiViewPatientDataset(
        records, transform=train_transform, image_size=int(config.get("image_size", 336))
    )
    val_ds = MultiViewPatientDataset(
        records, transform=val_transform, image_size=int(config.get("image_size", 336))
    )

    n_patients = len(train_ds)
    if n_patients == 0:
        raise ValueError("Dataset is empty. Check data_dir and CSV paths.")
    logger.info("Total patients in dataset: %d", n_patients)

    mgi_labels = np.array([s["labels"]["mgi"] for s in train_ds.samples])

    # -----------------------------------------------------------------------
    # Global class weight computation (across all patients)
    # -----------------------------------------------------------------------
    class_weights = {}
    num_classes_map = {
        "mgi": int(config.get("num_classes_mgi", 5)),
        "ohi": int(config.get("num_classes_ohi", 4)),
        "gei": int(config.get("num_classes_gei", 4)),
    }
    for task, n_cls in num_classes_map.items():
        task_labels = np.array([s["labels"][task] for s in train_ds.samples])
        class_weights[task] = compute_class_weights(task_labels, n_cls, device)
        logger.info(
            "Class weights [%s]: %s", task,
            [f"{w:.3f}" for w in class_weights[task].cpu().tolist()],
        )

    # -----------------------------------------------------------------------
    # K-Fold split at patient level
    # -----------------------------------------------------------------------
    n_folds = int(config["k_folds"])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(config.get("seed", 42)))
    all_indices = np.arange(n_patients)

    histories: List[List[Dict[str, Any]]] = []
    fold_summaries: List[Dict[str, Any]] = []
    checkpoint_paths: List[str] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, mgi_labels), start=1):
        logger.info("=" * 60)
        logger.info("FOLD %d / %d", fold_idx, n_folds)
        logger.info("  Train patients: %d  |  Val patients: %d", len(train_idx), len(val_idx))

        train_loader, val_loader = get_dataloaders(
            train_indices=train_idx,
            val_indices=val_idx,
            dataset=train_ds,
            val_dataset=val_ds,
            batch_size=int(config["batch_size"]),
            num_workers=int(config.get("num_workers", 0)),
            use_weighted_sampler=True,
            sampler_task="mgi",
        )

        # -----------------------------------------------------------------------
        # Model + Loss + Optimizer (Phase 1: frozen backbone)
        # -----------------------------------------------------------------------
        model = OralHealthModel(config=config).to(device)
        model.freeze_backbone()

        loss_fn = MultiTaskLoss(
            class_weights=class_weights,
            gamma_focal=float(config.get("focal_gamma", 2.0)),
            label_smoothing=float(config.get("label_smoothing", 0.1)),
            alpha_ce=float(config.get("alpha_ce", 1.0)),
            alpha_focal=float(config.get("alpha_focal", 0.5)),
            alpha_ordinal=float(config.get("alpha_ordinal", 0.3)),
            num_classes=num_classes_map,
        ).to(device)

        head_params = (
            list(model.shared_projection.parameters())
            + list(model.mgi_head.parameters())
            + list(model.ohi_head.parameters())
            + list(model.gei_head.parameters())
            + list(loss_fn.parameters())
        )
        optimizer = torch.optim.AdamW(
            head_params,
            lr=float(config["lr_heads"]),
            weight_decay=float(config.get("weight_decay", 1e-4)),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=int(config.get("scheduler_T0", 10)), eta_min=1e-7
        )

        scaler = amp.GradScaler() if device.type == "cuda" else None

        fold_ckpt = checkpoint_dir / f"checkpoint_fold_{fold_idx}.pth"
        stopper = EarlyStopping(
            patience=int(config["early_stopping_patience"]),
            checkpoint_path=fold_ckpt,
        )

        num_epochs = int(config["num_epochs"])
        freeze_epochs = int(config["freeze_epochs"])
        unfreeze_blocks = int(config.get("unfreeze_blocks", 4))

        fold_history: List[Dict[str, Any]] = []

        for epoch in range(1, num_epochs + 1):

            # Phase transition: unfreeze backbone after freeze_epochs
            if epoch == freeze_epochs + 1:
                model.unfreeze_last_n_blocks(unfreeze_blocks)
                param_groups = model.get_parameter_groups(
                    lr_backbone=float(config["lr_backbone"]),
                    lr_projection=float(config["lr_projection"]),
                    lr_heads=float(config["lr_heads"]),
                )
                param_groups.append({
                    "params": list(loss_fn.parameters()),
                    "lr": float(config["lr_heads"]),
                })
                optimizer = torch.optim.AdamW(
                    param_groups,
                    weight_decay=float(config.get("weight_decay", 1e-4)),
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=int(config.get("scheduler_T0", 10)), eta_min=1e-7
                )
                logger.info("Phase 2: backbone partially unfrozen (last %d blocks)", unfreeze_blocks)

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device,
                scaler=scaler,
                grad_clip=float(config.get("grad_clip_norm", 1.0)),
            )
            val_metrics = evaluate(
                model, val_loader, loss_fn, device,
                tta_steps=int(config.get("tta_steps", 1)) if config.get("tta_enabled") else 1,
            )
            scheduler.step()

            avg_f1 = float(np.mean([
                val_metrics["f1_macro"]["mgi"],
                val_metrics["f1_macro"]["ohi"],
                val_metrics["f1_macro"]["gei"],
            ]))

            row = {
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
                "val_f1_avg": avg_f1,
            }
            fold_history.append(row)

            logger.info(
                "Fold %d | Epoch %02d | train_loss=%.4f | val_f1(mgi/ohi/gei)=(%.3f/%.3f/%.3f) | avg_f1=%.3f",
                fold_idx, epoch,
                train_metrics["total_loss"],
                val_metrics["f1_macro"]["mgi"],
                val_metrics["f1_macro"]["ohi"],
                val_metrics["f1_macro"]["gei"],
                avg_f1,
            )

            should_stop = stopper.step(avg_f1, model, optimizer, loss_fn, epoch, config)
            if should_stop:
                logger.info("Early stopping at epoch %d (best val_f1=%.4f)", epoch, stopper.best_f1)
                break

        histories.append(fold_history)
        checkpoint_paths.append(str(fold_ckpt))

        # Reload best checkpoint for final evaluation
        if fold_ckpt.exists():
            best_ckpt = torch.load(fold_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state_dict"])
        final_val = evaluate(model, val_loader, loss_fn, device)

        fold_summaries.append({
            "fold": fold_idx,
            "best_val_f1": stopper.best_f1,
            "checkpoint": str(fold_ckpt),
            "final_val_accuracy": final_val["accuracy"],
            "final_val_f1": final_val["f1_macro"],
        })

        logger.info(
            "Fold %d best → val_f1_avg=%.4f | acc(mgi/ohi/gei)=(%.3f/%.3f/%.3f)",
            fold_idx, stopper.best_f1,
            final_val["accuracy"]["mgi"],
            final_val["accuracy"]["ohi"],
            final_val["accuracy"]["gei"],
        )

    # -----------------------------------------------------------------------
    # Save ensemble config
    # -----------------------------------------------------------------------
    avg_f1 = {
        task: float(np.mean([f["final_val_f1"][task] for f in fold_summaries]))
        for task in ("mgi", "ohi", "gei")
    }

    ensemble_config = {
        "models": checkpoint_paths,
        "weights": [1.0] * len(checkpoint_paths),
        "config": config,
        "avg_val_f1": avg_f1,
    }

    # Save in both checkpoints dir and models/ root
    ckpt_ensemble = checkpoint_dir / "ensemble_config.json"
    models_root = checkpoint_dir.parent
    models_ensemble = models_root / "ensemble_config.json"

    for path in (ckpt_ensemble, models_ensemble):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(ensemble_config, f, indent=2)
        logger.info("Ensemble config saved → %s", path)

    logger.info("=" * 60)
    logger.info("K-Fold training complete. Average val F1: %s", avg_f1)

    return {
        "histories": histories,
        "fold_summaries": fold_summaries,
        "avg_val_f1": avg_f1,
        "checkpoint_paths": checkpoint_paths,
    }
