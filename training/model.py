"""Multi-task DINOv2 model for oral health score classification."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import timm


class OralHealthModel(nn.Module):
    """Multi-task classifier with DINOv2 backbone and task-specific heads.

    Architecture:
        Backbone -> shared projection -> three heads (MGI, OHI, GEI)
    """

    def __init__(self, config: Dict[str, object]) -> None:
        """Initialize the model.

        Args:
            config: Configuration dictionary containing backbone name,
                dropout, and class counts.
        """
        super().__init__()
        backbone_name = str(config["backbone_model"])
        dropout = float(config["dropout"])

        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        backbone_dim = int(getattr(self.backbone, "num_features"))

        self.shared_projection = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mgi_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, int(config["num_classes_mgi"])),
        )
        self.ohi_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, int(config["num_classes_ohi"])),
        )
        self.gei_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, int(config["num_classes_gei"])),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input patch tensor of shape (B, C, H, W).

        Returns:
            Dictionary of logits for each task.
        """
        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[0]

        shared = self.shared_projection(features)
        return {
            "mgi": self.mgi_head(shared),
            "ohi": self.ohi_head(shared),
            "gei": self.gei_head(shared),
        }

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze the last n transformer blocks of the backbone.

        Args:
            n: Number of final backbone blocks to unfreeze.
        """
        # Freeze everything first.
        self.freeze_backbone()

        if n <= 0:
            return

        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            # If backbone has no .blocks attr, unfreeze all backbone params.
            for param in self.backbone.parameters():
                param.requires_grad = True
            return

        total = len(blocks)
        start_idx = max(0, total - n)
        for idx in range(start_idx, total):
            for param in blocks[idx].parameters():
                param.requires_grad = True

        # Unfreeze final normalization layers when present.
        for norm_name in ("norm", "fc_norm"):
            norm_module = getattr(self.backbone, norm_name, None)
            if norm_module is not None:
                for param in norm_module.parameters():
                    param.requires_grad = True
