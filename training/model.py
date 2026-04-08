"""Multi-task DINOv2 model for oral health score classification.

Architecture:
    DINOv2 backbone (shared) → 3-view feature concatenation
    → shared projection head → three task-specific heads (MGI, OHI, GEI)

The shared DINOv2 backbone is loaded with pretrained weights.
Training strategy:
    Phase 1 (freeze_epochs): Only train projection + task heads.
    Phase 2 (unfreeze last N blocks): Fine-tune backbone at very low LR.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)


class _TaskHead(nn.Module):
    """Three-layer MLP head for one classification task.

    Uses LayerNorm (not BatchNorm) so it works correctly with small batch sizes.

    Args:
        in_features: Dimension of input feature vector.
        num_classes: Number of output classes.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.45,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class OralHealthModel(nn.Module):
    """Multi-view multi-task dental index classifier.

    Accepts three views (frontal, left, right) processed through a shared
    DINOv2-small backbone, concatenates their feature vectors, and feeds the
    fused representation through task-specific heads.

    Args:
        config: Configuration dictionary.  Must contain:
            backbone_model, dropout, num_classes_mgi, num_classes_ohi,
            num_classes_gei, projection_dim, head_hidden_dim.
    """

    # DINOv2-small feature dimension
    _BACKBONE_DIM = 384

    def __init__(self, config: Dict) -> None:
        super().__init__()
        backbone_name = str(config.get("backbone_model", "vit_small_patch14_dinov2.lvd142m"))
        dropout = float(config.get("dropout", 0.45))
        proj_dim = int(config.get("projection_dim", 512))
        hidden_dim = int(config.get("head_hidden_dim", 256))

        # -------------------------------------------------------
        # Backbone (single shared instance, applied to each view)
        # -------------------------------------------------------
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        backbone_dim = int(getattr(self.backbone, "num_features", self._BACKBONE_DIM))

        # -------------------------------------------------------
        # Shared projection: fused 3-view features → projection_dim
        # -------------------------------------------------------
        fused_dim = backbone_dim * 3  # concatenate frontal + left + right
        self.shared_projection = nn.Sequential(
            nn.Linear(fused_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # -------------------------------------------------------
        # Task-specific heads
        # -------------------------------------------------------
        n_mgi = int(config.get("num_classes_mgi", 5))
        n_ohi = int(config.get("num_classes_ohi", 4))
        n_gei = int(config.get("num_classes_gei", 4))

        self.mgi_head = _TaskHead(proj_dim, n_mgi, hidden_dim, dropout)
        self.ohi_head = _TaskHead(proj_dim, n_ohi, hidden_dim, dropout)
        self.gei_head = _TaskHead(proj_dim, n_gei, hidden_dim, dropout)

        logger.info(
            "OralHealthModel built: backbone=%s fused_dim=%d proj_dim=%d",
            backbone_name, fused_dim, proj_dim,
        )

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone on a single view tensor.

        Args:
            x: Image tensor (B, C, H, W).

        Returns:
            Feature tensor (B, backbone_dim).
        """
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[0]
        return features

    def forward(
        self,
        frontal: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Multi-view forward pass.

        Args:
            frontal: Frontal image batch (B, C, H, W).
            left: Left lateral image batch (B, C, H, W).
            right: Right lateral image batch (B, C, H, W).

        Returns:
            Dict with logit tensors for 'mgi', 'ohi', 'gei' tasks.
        """
        f_feat = self._extract_features(frontal)
        l_feat = self._extract_features(left)
        r_feat = self._extract_features(right)

        fused = torch.cat([f_feat, l_feat, r_feat], dim=1)
        shared = self.shared_projection(fused)

        return {
            "mgi": self.mgi_head(shared),
            "ohi": self.ohi_head(shared),
            "gei": self.gei_head(shared),
        }

    # ----------------------------------------------------------
    # Backbone freezing controls
    # ----------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen.")

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze the last n transformer blocks of the backbone.

        Args:
            n: Number of final blocks to unfreeze. 0 = keep all frozen.
        """
        # First freeze everything, then selectively unfreeze
        self.freeze_backbone()

        if n <= 0:
            return

        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            # Fallback: unfreeze the entire backbone if no .blocks attr
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.warning("Backbone has no .blocks attr — unfreezing all backbone params.")
            return

        total = len(blocks)
        start_idx = max(0, total - n)
        for idx in range(start_idx, total):
            for param in blocks[idx].parameters():
                param.requires_grad = True

        # Unfreeze final normalization layers
        for norm_name in ("norm", "fc_norm"):
            norm_mod = getattr(self.backbone, norm_name, None)
            if norm_mod is not None:
                for param in norm_mod.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        logger.info("Unfroze last %d backbone blocks. Trainable backbone params: %d", n, trainable)

    def get_parameter_groups(
        self,
        lr_backbone: float,
        lr_projection: float,
        lr_heads: float,
    ) -> List[Dict]:
        """Build parameter groups with differential learning rates.

        Args:
            lr_backbone: Learning rate for backbone parameters.
            lr_projection: Learning rate for shared projection.
            lr_heads: Learning rate for task heads.

        Returns:
            List suitable for passing to torch.optim.AdamW.
        """
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        proj_params = list(self.shared_projection.parameters())
        head_params = (
            list(self.mgi_head.parameters())
            + list(self.ohi_head.parameters())
            + list(self.gei_head.parameters())
        )

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        groups.append({"params": proj_params, "lr": lr_projection})
        groups.append({"params": head_params, "lr": lr_heads})
        return groups

    # ----------------------------------------------------------
    # Inference helper
    # ----------------------------------------------------------

    def predict_scores(
        self,
        frontal: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Dict[str, Dict]:
        """Predict scores with softmax confidence.

        Args:
            frontal, left, right: Image tensors (B, C, H, W) on correct device.

        Returns:
            Dict keyed by task with sub-dicts: score, confidence, probs.
        """
        self.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=frontal.is_cuda):
                outputs = self.forward(frontal, left, right)

        results = {}
        for key in ("mgi", "ohi", "gei"):
            probs = torch.softmax(outputs[key].float(), dim=1)
            predicted = probs.argmax(dim=1).int()
            confidence = probs.max(dim=1).values * 100.0
            results[key] = {
                "score": predicted,
                "confidence": confidence,
                "probs": probs,
            }
        return results
