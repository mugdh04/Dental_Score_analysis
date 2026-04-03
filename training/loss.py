"""Loss functions for multi-task oral health classification."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Uncertainty-weighted multi-task cross-entropy loss.

    Implements Kendall et al. uncertainty-based task weighting with learnable
    log-variance parameters for MGI, OHI, and GEI tasks.
    """

    def __init__(self) -> None:
        """Initialize per-task CE losses and learnable log variances."""
        super().__init__()
        self.ce_mgi = nn.CrossEntropyLoss()
        self.ce_ohi = nn.CrossEntropyLoss()
        self.ce_gei = nn.CrossEntropyLoss()

        self.log_var_mgi = nn.Parameter(torch.zeros(1))
        self.log_var_ohi = nn.Parameter(torch.zeros(1))
        self.log_var_gei = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _weighted(loss: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Apply uncertainty weighting to a task loss."""
        precision = torch.exp(-log_var)
        return 0.5 * precision * loss + log_var

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute weighted total loss and individual task losses.

        Args:
            predictions: Dict with logits for keys mgi/ohi/gei.
            targets: Dict with integer class labels for keys mgi/ohi/gei.

        Returns:
            Tuple of (total_loss, individual_losses_dict).
        """
        mgi_loss = self.ce_mgi(predictions["mgi"], targets["mgi"])
        ohi_loss = self.ce_ohi(predictions["ohi"], targets["ohi"])
        gei_loss = self.ce_gei(predictions["gei"], targets["gei"])

        total = (
            self._weighted(mgi_loss, self.log_var_mgi)
            + self._weighted(ohi_loss, self.log_var_ohi)
            + self._weighted(gei_loss, self.log_var_gei)
        )

        return total.squeeze(), {
            "mgi": mgi_loss,
            "ohi": ohi_loss,
            "gei": gei_loss,
        }
