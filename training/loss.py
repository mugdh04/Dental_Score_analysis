"""Loss functions for multi-task oral health classification.

Implements a combined loss strategy to aggressively combat class imbalance
and the 'predict-only-extremes' failure mode:

1. Class-weighted CrossEntropyLoss  — up-weights minority class errors
2. Focal Loss                        — down-weights easy majority-class samples
3. Ordinal Loss                      — penalises predictions far from truth on index scale
4. Uncertainty weighting (Kendall)   — learnable per-task scaling
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Focal Loss helper
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017).

    Reduces the relative loss for well-classified examples so the model
    focuses on hard, minority-class samples.

    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
        weight: Optional per-class weight tensor (same as CrossEntropyLoss weight).
        label_smoothing: Label smoothing factor.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model outputs of shape (B, C).
            targets: Ground-truth integer class indices of shape (B,).

        Returns:
            Scalar focal loss tensor.
        """
        weight = self.weight if hasattr(self, "weight") and self.weight is not None else None

        # Standard cross-entropy with reduction='none' to apply focal weighting
        ce = F.cross_entropy(
            logits,
            targets,
            weight=weight.to(logits.device) if weight is not None else None,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)  # probability of correct class
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ---------------------------------------------------------------------------
# Ordinal Loss helper
# ---------------------------------------------------------------------------

class OrdinalLoss(nn.Module):
    """Ordinal regression loss for ordered class indices.

    Treats the class index as a numeric scale (e.g., MGI 0-4) and imposes a
    penalty proportional to the absolute difference between predicted and true
    class, in addition to standard CE.  This prevents the model from being
    indifferent between adjacent and distant class errors.

    Args:
        num_classes: Number of ordinal classes.
        weight: Optional per-class weight tensor.
    """

    def __init__(self, num_classes: int, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ordinal loss.

        Args:
            logits: Raw model outputs (B, C).
            targets: Ground-truth integer indices (B,).

        Returns:
            Scalar ordinal loss.
        """
        probs = F.softmax(logits, dim=1)  # (B, C)
        # Build an ordinal index tensor [0, 1, ..., C-1] — same device as logits
        ordinal_indices = torch.arange(self.num_classes, device=logits.device).float()  # (C,)

        # Expected predicted class value
        expected = (probs * ordinal_indices.unsqueeze(0)).sum(dim=1)  # (B,)
        true_vals = targets.float()

        # MSE between predicted expected value and true ordinal value
        loss = F.mse_loss(expected, true_vals)
        return loss


# ---------------------------------------------------------------------------
# Combined Multi-Task Loss
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """Combined uncertainty-weighted multi-task loss.

    For each task (MGI, OHI, GEI) computes:
        total_task_loss = α * weighted_CE + β * FocalLoss + γ * OrdinalLoss

    All three components are summed with learnable Kendall uncertainty weights.

    Args:
        class_weights:  Dict mapping task name → per-class weight tensor.
                        Pass result of sklearn.utils.compute_class_weight.
        gamma_focal:    Focal loss γ parameter.
        label_smoothing: Label smoothing applied to CE and FocalLoss.
        alpha_ce:       Weight of CrossEntropy component.
        alpha_focal:    Weight of Focal component.
        alpha_ordinal:  Weight of Ordinal component.
        num_classes:    Dict mapping task name → number of classes.
    """

    def __init__(
        self,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        gamma_focal: float = 2.0,
        label_smoothing: float = 0.1,
        alpha_ce: float = 1.0,
        alpha_focal: float = 0.5,
        alpha_ordinal: float = 0.3,
        num_classes: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()

        if num_classes is None:
            num_classes = {"mgi": 5, "ohi": 4, "gei": 4}

        cw = class_weights or {}

        # CrossEntropyLoss — with class weights + label smoothing
        self.ce_mgi = nn.CrossEntropyLoss(
            weight=cw.get("mgi"), label_smoothing=label_smoothing
        )
        self.ce_ohi = nn.CrossEntropyLoss(
            weight=cw.get("ohi"), label_smoothing=label_smoothing
        )
        self.ce_gei = nn.CrossEntropyLoss(
            weight=cw.get("gei"), label_smoothing=label_smoothing
        )

        # Focal Loss — with class weights + label smoothing
        self.focal_mgi = FocalLoss(
            gamma=gamma_focal, weight=cw.get("mgi"), label_smoothing=label_smoothing
        )
        self.focal_ohi = FocalLoss(
            gamma=gamma_focal, weight=cw.get("ohi"), label_smoothing=label_smoothing
        )
        self.focal_gei = FocalLoss(
            gamma=gamma_focal, weight=cw.get("gei"), label_smoothing=label_smoothing
        )

        # Ordinal Loss
        self.ordinal_mgi = OrdinalLoss(num_classes["mgi"], weight=cw.get("mgi"))
        self.ordinal_ohi = OrdinalLoss(num_classes["ohi"], weight=cw.get("ohi"))
        self.ordinal_gei = OrdinalLoss(num_classes["gei"], weight=cw.get("gei"))

        # Learnable Kendall log-variances (one per task)
        self.log_var_mgi = nn.Parameter(torch.zeros(1))
        self.log_var_ohi = nn.Parameter(torch.zeros(1))
        self.log_var_gei = nn.Parameter(torch.zeros(1))

        self.alpha_ce = alpha_ce
        self.alpha_focal = alpha_focal
        self.alpha_ordinal = alpha_ordinal

    # ----------------------------------------------------------
    def _task_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ce: nn.CrossEntropyLoss,
        focal: FocalLoss,
        ordinal: OrdinalLoss,
        log_var: nn.Parameter,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute combined + uncertainty-weighted loss for one task.

        Returns:
            (uncertainty_weighted_total, raw_task_loss_for_logging)
        """
        # Move class-weight buffers to the target device (handles CPU/GPU switch)
        device = logits.device
        for criterion in (ce, focal, ordinal):
            if hasattr(criterion, "weight") and criterion.weight is not None:
                criterion.weight = criterion.weight.to(device)

        l_ce = ce(logits, targets)
        l_focal = focal(logits, targets)
        l_ordinal = ordinal(logits, targets)

        raw = self.alpha_ce * l_ce + self.alpha_focal * l_focal + self.alpha_ordinal * l_ordinal

        # Kendall uncertainty weighting: loss / (2*sigma^2) + log(sigma)
        precision = torch.exp(-log_var)
        weighted = 0.5 * precision * raw + log_var

        return weighted.squeeze(), raw

    # ----------------------------------------------------------
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss and per-task losses.

        Args:
            predictions: Dict of logits keyed by task name.
            targets: Dict of integer label tensors keyed by task name.

        Returns:
            Tuple (total_loss, {mgi: raw_loss, ohi: raw_loss, gei: raw_loss}).
        """
        w_mgi, r_mgi = self._task_loss(
            predictions["mgi"], targets["mgi"],
            self.ce_mgi, self.focal_mgi, self.ordinal_mgi, self.log_var_mgi,
        )
        w_ohi, r_ohi = self._task_loss(
            predictions["ohi"], targets["ohi"],
            self.ce_ohi, self.focal_ohi, self.ordinal_ohi, self.log_var_ohi,
        )
        w_gei, r_gei = self._task_loss(
            predictions["gei"], targets["gei"],
            self.ce_gei, self.focal_gei, self.ordinal_gei, self.log_var_gei,
        )

        total = w_mgi + w_ohi + w_gei

        return total, {
            "mgi": r_mgi,
            "ohi": r_ohi,
            "gei": r_gei,
        }
