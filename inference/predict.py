"""Inference module — ensemble prediction with multi-view dental model.

Supports two model formats:
  1. NEW: Multi-view OralHealthModel (training/train_model.ipynb output)
          stored as: models/ensemble_config.json → models/checkpoints/checkpoint_fold_N.pth
  2. LEGACY: EnsembleDentalModel (ml/ system)
             stored as: ml/checkpoints/fold_N_head.pth + ml/checkpoints/best_model.pth

The predictor automatically detects which format is present and dispatches
accordingly so the backend does not need to change.

Change note (2026-04-08):
  Updated to support new multi-view OralHealthModel architecture trained with
  weighted-CE + focal + ordinal loss. Maintains backward compat with ml/ models.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

ThresholdValue = Union[float, List[float], Tuple[float, ...]]


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _coerce_threshold_value(value: Any, default: float = 0.5) -> ThresholdValue:
    if _is_sequence(value):
        seq = [float(v) for v in list(value)]
        return seq if seq else default
    try:
        return float(value)
    except Exception:
        return default


def _format_thresholds(value: ThresholdValue) -> str:
    if _is_sequence(value):
        return "[" + ",".join(f"{float(v):.3f}" for v in list(value)) + "]"
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "0.500"


def _ordinal_to_class_probs(ordinal_probs: torch.Tensor) -> torch.Tensor:
    """Convert cumulative ordinal probabilities to class probabilities."""
    bsz = ordinal_probs.shape[0]
    ones = torch.ones(bsz, 1, device=ordinal_probs.device, dtype=ordinal_probs.dtype)
    zeros = torch.zeros(bsz, 1, device=ordinal_probs.device, dtype=ordinal_probs.dtype)
    cum_full = torch.cat([ones, ordinal_probs, zeros], dim=1)
    class_probs = (cum_full[:, :-1] - cum_full[:, 1:]).clamp(min=1e-8)
    return class_probs / class_probs.sum(dim=1, keepdim=True)


def _decode_ordinal_probs(
    probs: torch.Tensor,
    thresholds: ThresholdValue,
    task: str = "",
) -> torch.Tensor:
    if thresholds is None:
        thresholds = 0.5
    if _is_sequence(thresholds):
        thr_list = [float(v) for v in list(thresholds)]
        if len(thr_list) != probs.shape[1]:
            fallback = float(np.mean(thr_list)) if thr_list else 0.5
            logger.warning(
                "Threshold length mismatch for %s (got %d, expected %d); using scalar %.3f",
                task,
                len(thr_list),
                probs.shape[1],
                fallback,
            )
            return (probs > fallback).sum(dim=1).long()
        thr_t = torch.tensor(thr_list, dtype=probs.dtype, device=probs.device)
        return (probs > thr_t.unsqueeze(0)).sum(dim=1).long()
    try:
        thr = float(thresholds)
    except Exception:
        thr = 0.5
    return (probs > thr).sum(dim=1).long()


def _decode_ordinal_probs_np(
    probs: np.ndarray,
    thresholds: ThresholdValue,
    task: str = "",
) -> int:
    p = np.asarray(probs, dtype=float).reshape(-1)
    if thresholds is None:
        thresholds = 0.5
    if _is_sequence(thresholds):
        thr_list = [float(v) for v in list(thresholds)]
        if len(thr_list) != p.size:
            fallback = float(np.mean(thr_list)) if thr_list else 0.5
            logger.warning(
                "Threshold length mismatch for %s (got %d, expected %d); using scalar %.3f",
                task,
                len(thr_list),
                p.size,
                fallback,
            )
            return int((p > fallback).sum())
        thr = np.asarray(thr_list, dtype=float)
        return int((p > thr).sum())
    try:
        thr = float(thresholds)
    except Exception:
        thr = 0.5
    return int((p > thr).sum())

# ---------------------------------------------------------------------------
# Internal: Multi-view OralHealthModel (matches training/model.py)
# ---------------------------------------------------------------------------

class _TaskHead(nn.Module):
    """Two-layer MLP head.  Output dim = num_classes - 1 for ordinal BCE."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def _create_backbone_eval(name: str, img_size: int) -> nn.Module:
    import timm
    if "efficientnet" in name.lower() or "resnet" in name.lower():
        return timm.create_model(name, pretrained=False, num_classes=0, global_pool="avg")
    return timm.create_model(name, pretrained=False, img_size=img_size, num_classes=0, global_pool="avg")


def _create_dino_tokens(name: str, img_size: int) -> nn.Module:
    """Create DINOv2 backbone that returns token sequences (global_pool="")."""
    import timm
    return timm.create_model(name, pretrained=False, img_size=img_size, num_classes=0, global_pool="")


def _extract_dino_features(backbone: nn.Module, x: torch.Tensor, use_cls: bool) -> torch.Tensor:
    """Extract CLS+patch-avg or patch-avg features from a DINOv2 backbone."""
    out = backbone(x)
    if out.dim() == 2:
        return out
    cls_tok = out[:, 0, :]
    patch_avg = out[:, 1:, :].mean(1)
    if use_cls:
        return torch.cat([cls_tok, patch_avg], dim=1)
    return patch_avg

class ViewAttentionPool(nn.Module):
    """Self-attention over the 3 intra-oral views, then mean-pool."""
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(views, views, views)
        views = self.norm1(views + attn_out)
        ff_out = self.ff(views)
        views  = self.norm2(views + ff_out)
        return views.mean(dim=1)

class _OralHealthModel(nn.Module):
    """Inference model supporting both v2 and v3 checkpoint architectures."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        dropout  = float(config.get("dropout", 0.20))
        proj_dim = int(config.get("projection_dim", 256))
        head_dim = int(config.get("head_hidden_dim", 128))
        img_size = int(config.get("image_size", 336))

        self.is_v3 = ("backbone_dino" in config) or ("backbone_cnn" in config)

        if self.is_v3:
            dino_name = config.get("backbone_dino", "vit_small_patch14_dinov2.lvd142m")
            cnn_name  = config.get("backbone_cnn", "efficientnet_b4")
            self.dino = _create_backbone_eval(dino_name, img_size)
            self.cnn  = _create_backbone_eval(cnn_name, img_size)

            dino_dim = self.dino.num_features
            cnn_dim  = self.cnn.num_features
            fused_dim = dino_dim + cnn_dim

            self.view_pool = ViewAttentionPool(
                dino_dim, num_heads=config.get("view_attn_heads", 4)
            )
            self.projection = nn.Sequential(
                nn.Linear(fused_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.mgi_head = _TaskHead(proj_dim, config.get("num_classes_mgi", 5) - 1, head_dim, dropout)
            self.ohi_head = _TaskHead(proj_dim, config.get("num_classes_ohi", 3) - 1, head_dim, dropout)
            self.gei_head = _TaskHead(proj_dim, config.get("num_classes_gei", 3) - 1, head_dim, dropout)
            self.gei_aux  = _TaskHead(proj_dim, 1, 64, dropout)
        else:
            backbone_name = config.get("backbone_model", "vit_small_patch14_dinov2.lvd142m")
            self.backbone = _create_backbone_eval(backbone_name, img_size)
            feat_dim = self.backbone.num_features

            self.view_pool = ViewAttentionPool(
                feat_dim, num_heads=config.get("view_attn_heads", 4)
            )
            self.projection = nn.Sequential(
                nn.Linear(feat_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.mgi_head = _TaskHead(proj_dim, config.get("num_classes_mgi", 5) - 1, head_dim, dropout)
            self.ohi_head = _TaskHead(proj_dim, config.get("num_classes_ohi", 3) - 1, head_dim, dropout)
            self.gei_head = _TaskHead(proj_dim, config.get("num_classes_gei", 3) - 1, head_dim, dropout)

    @staticmethod
    def _extract_feature(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = backbone(x)
        return out[:, 0] if out.dim() == 3 else out

    def forward(self, frontal: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.is_v3:
            fd = self._extract_feature(self.dino, frontal)
            ld = self._extract_feature(self.dino, left)
            rd = self._extract_feature(self.dino, right)
            dino_feat = self.view_pool(torch.stack([fd, ld, rd], dim=1))

            fc = self._extract_feature(self.cnn, frontal)
            lc = self._extract_feature(self.cnn, left)
            rc = self._extract_feature(self.cnn, right)
            cnn_feat = (fc + lc + rc) / 3.0

            shared = self.projection(torch.cat([dino_feat, cnn_feat], dim=1))
            return {
                "mgi": self.mgi_head(shared),
                "ohi": self.ohi_head(shared),
                "gei": self.gei_head(shared),
                "gei_aux": self.gei_aux(shared),
            }

        f = self._extract_feature(self.backbone, frontal)
        l = self._extract_feature(self.backbone, left)
        r = self._extract_feature(self.backbone, right)

        views = torch.stack([f, l, r], dim=1)
        pooled = self.view_pool(views)
        shared = self.projection(pooled)

        return {
            "mgi": self.mgi_head(shared),
            "ohi": self.ohi_head(shared),
            "gei": self.gei_head(shared),
        }

    def predict_scores(
        self,
        frontal,
        left,
        right,
        thresholds: Optional[Dict[str, ThresholdValue]] = None,
        temperature: float = 1.0,
    ):
        self.eval()
        thresholds = thresholds or {}
        temp = max(float(temperature), 1e-4)
        with torch.no_grad():
            outputs = self.forward(frontal, left, right)

        result = {}
        for key in ("mgi", "ohi", "gei"):
            logits = outputs[key].float()
            ord_probs = torch.sigmoid(logits / temp)
            thr = thresholds.get(key, 0.5)
            predicted = _decode_ordinal_probs(ord_probs, thr, task=key)

            class_probs = _ordinal_to_class_probs(ord_probs)
            
            confidence = class_probs.max(dim=1).values * 100.0

            result[key] = {
                "score": predicted,
                "confidence": confidence,
                "probs": class_probs,
                "ordinal_probs": ord_probs,
            }
        return result


class _OralHealthModelV4(nn.Module):
    """Inference model aligned with training/train_model_v4.py."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.use_cls = bool(config.get("dino_use_cls", True))
        self.cnn_attn = bool(config.get("cnn_cross_view_attn", True))

        dropout = float(config.get("dropout", 0.25))
        proj_dim = int(config.get("projection_dim", 256))
        head_dim = int(config.get("head_hidden_dim", 128))
        img_size = int(config.get("image_size", 336))

        dino_name = config.get("backbone_dino", "vit_small_patch14_dinov2.lvd142m")
        cnn_name = config.get("backbone_cnn", "efficientnet_b4")

        self.dino = _create_dino_tokens(dino_name, img_size)
        self.cnn = _create_backbone_eval(cnn_name, img_size)

        dino_base = int(getattr(self.dino, "num_features", 384))
        dino_dim = dino_base * 2 if self.use_cls else dino_base
        cnn_dim = int(getattr(self.cnn, "num_features", 1792))
        fused_dim = dino_dim + cnn_dim

        self.dino_pool = ViewAttentionPool(
            dino_dim, num_heads=config.get("view_attn_heads", 4)
        )
        if self.cnn_attn:
            self.cnn_pool = ViewAttentionPool(
                cnn_dim, num_heads=config.get("view_attn_heads", 4)
            )

        self.proj = nn.Sequential(
            nn.Linear(fused_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        sc_dim = int(config.get("supcon_dim", 128))
        for t in ("mgi", "ohi", "gei"):
            setattr(
                self,
                f"sc_proj_{t}",
                nn.Sequential(
                    nn.Linear(proj_dim, sc_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(sc_dim, sc_dim),
                ),
            )

        self.mgi_head = _TaskHead(proj_dim, config.get("num_classes_mgi", 5) - 1, head_dim, dropout)
        self.ohi_head = _TaskHead(proj_dim, config.get("num_classes_ohi", 3) - 1, head_dim, dropout)
        self.gei_head = _TaskHead(proj_dim, config.get("num_classes_gei", 3) - 1, head_dim, dropout)
        self.gei_aux = _TaskHead(proj_dim, 1, 64, dropout)

    def _dino_feat(self, x: torch.Tensor) -> torch.Tensor:
        return _extract_dino_features(self.dino, x, self.use_cls)

    def _cnn_feat(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)

    def _fuse_views(self, frontal, left, right) -> torch.Tensor:
        d_stack = torch.stack([
            self._dino_feat(frontal),
            self._dino_feat(left),
            self._dino_feat(right),
        ], dim=1)
        d_out = self.dino_pool(d_stack)

        c_stack = torch.stack([
            self._cnn_feat(frontal),
            self._cnn_feat(left),
            self._cnn_feat(right),
        ], dim=1)
        c_out = self.cnn_pool(c_stack) if self.cnn_attn else c_stack.mean(dim=1)

        return torch.cat([d_out, c_out], dim=1)

    def forward(self, frontal, left, right, return_features: bool = False):
        fused = self._fuse_views(frontal, left, right)
        proj = self.proj(fused)

        out = {
            "mgi": self.mgi_head(proj),
            "ohi": self.ohi_head(proj),
            "gei": self.gei_head(proj),
            "gei_aux": self.gei_aux(proj),
        }

        if return_features:
            out["features"] = {
                t: F.normalize(getattr(self, f"sc_proj_{t}")(proj), dim=1)
                for t in ("mgi", "ohi", "gei")
            }

        return out

    def predict_scores(
        self,
        frontal,
        left,
        right,
        thresholds: Optional[Dict[str, ThresholdValue]] = None,
        temperature: float = 1.0,
    ):
        self.eval()
        thresholds = thresholds or {}
        temp = max(float(temperature), 1e-4)
        with torch.no_grad():
            outputs = self.forward(frontal, left, right)

        result = {}
        for key in ("mgi", "ohi", "gei"):
            logits = outputs[key].float()
            ord_probs = torch.sigmoid(logits / temp)
            thr = thresholds.get(key, 0.5)
            predicted = _decode_ordinal_probs(ord_probs, thr, task=key)

            class_probs = _ordinal_to_class_probs(ord_probs)
            confidence = class_probs.max(dim=1).values * 100.0

            result[key] = {
                "score": predicted,
                "confidence": confidence,
                "probs": class_probs,
                "ordinal_probs": ord_probs,
            }
        return result


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------

def _standardize_image(image_path: str, target_size: int) -> np.ndarray:
    """Load, resize, white-balance, and CLAHE-enhance an image.

    Args:
        image_path: Path to source image file.
        target_size: Square output size.

    Returns:
        RGB uint8 array of shape (target_size, target_size, 3).

    Raises:
        ValueError: If image cannot be loaded.
    """
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img_bgr = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Gray-world white balance
    avg = img_rgb.mean(axis=(0, 1)).astype(np.float32)
    gray_avg = avg.mean()
    scale = np.where(avg > 1e-6, gray_avg / avg, 1.0)
    img_wb = np.clip(img_rgb.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    # CLAHE on L channel
    lab = cv2.cvtColor(img_wb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_out = cv2.cvtColor(cv2.merge((clahe.apply(l_ch), a_ch, b_ch)), cv2.COLOR_LAB2RGB)
    return img_out


# ---------------------------------------------------------------------------
# Task score labels
# ---------------------------------------------------------------------------

_MGI_LABELS = {0: "No inflammation", 1: "Mild inflammation", 2: "Moderate inflammation",
               3: "Severe inflammation", 4: "Severe with spontaneous bleeding"}
_OHI_LABELS = {0: "Good oral hygiene", 1: "Fair oral hygiene",
               2: "Poor oral hygiene", 3: "Very poor oral hygiene"}
_GEI_LABELS = {0: "No enlargement", 1: "Mild enlargement",
               2: "Moderate enlargement", 3: "Severe enlargement"}
_TASK_LABELS = {"mgi": _MGI_LABELS, "ohi": _OHI_LABELS, "gei": _GEI_LABELS}


# ---------------------------------------------------------------------------
# OralHealthPredictor — primary inference class
# ---------------------------------------------------------------------------

class OralHealthPredictor:
    """Multi-view dental index predictor with ensemble support.

    Supports both new multi-view OralHealthModel checkpoints and legacy
    ml/ EnsembleDentalModel checkpoints.

    Args:
        ensemble_config_path: Path to ensemble_config.json.
        pi_calibration_path: Path to pi_calibration.json.
        device: Torch device string ('cpu' or 'cuda').
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        ensemble_config_path: str,
        pi_calibration_path: str,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.models: List[nn.Module] = []
        self.model_config: Dict = {}
        self.decode_thresholds: Dict[str, ThresholdValue] = {
            "mgi": 0.5,
            "ohi": 0.5,
            "gei": 0.5,
        }
        self.decode_temperature: float = 1.0
        self.image_size: int = 336
        self._last_latency: float = 0.0

        self._load_models(ensemble_config_path)
        self._load_pi_calibration(pi_calibration_path)

        self.val_transform = A.Compose([
            A.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ToTensorV2(),
        ])

        self._warmup()

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def _load_models(self, ensemble_config_path: str) -> None:
        """Load all fold models according to ensemble_config.json.

        If new multi-view format is detected (config dict embedded in checkpoint),
        loads _OralHealthModel. Otherwise falls back to legacy ml/ model loading.
        """
        ensemble_path = Path(ensemble_config_path)
        if not ensemble_path.exists():
            logger.error("ensemble_config.json not found at %s. Inference unavailable.", ensemble_config_path)
            return

        with open(ensemble_path) as f:
            ec = json.load(f)

        thr_cfg = ec.get("avg_thresholds") or ec.get("decode_thresholds_global") or {}
        self.decode_thresholds = {
            task: _coerce_threshold_value(thr_cfg.get(task, 0.5))
            for task in ("mgi", "ohi", "gei")
        }
        self.decode_temperature = float(ec.get("avg_temperature", 1.0))
        logger.info(
            "Decode thresholds: mgi=%s ohi=%s gei=%s (temp=%.3f)",
            _format_thresholds(self.decode_thresholds["mgi"]),
            _format_thresholds(self.decode_thresholds["ohi"]),
            _format_thresholds(self.decode_thresholds["gei"]),
            self.decode_temperature,
        )

        model_type = str(ec.get("model_type", "")).lower()
        raw_config = ec.get("config")
        use_v4 = "oralhealthmodelv4" in model_type or model_type.endswith("v4")

        default_v3 = {
            "backbone_model": "vit_small_patch14_dinov2.lvd142m",
            "dropout": 0.45,
            "projection_dim": 512,
            "head_hidden_dim": 256,
            "num_classes_mgi": 5,
            "num_classes_ohi": 4,
            "num_classes_gei": 4,
            "image_size": 336,
        }
        default_v4 = {
            "backbone_dino": "vit_small_patch14_dinov2.lvd142m",
            "backbone_cnn": "efficientnet_b4",
            "pretrained": False,
            "dropout": 0.25,
            "projection_dim": 256,
            "head_hidden_dim": 128,
            "num_classes_mgi": 5,
            "num_classes_ohi": 3,
            "num_classes_gei": 3,
            "image_size": 336,
            "view_attn_heads": 4,
            "dino_use_cls": True,
            "cnn_cross_view_attn": True,
            "supcon_dim": 128,
        }

        if raw_config is None:
            self.model_config = default_v4 if use_v4 else default_v3
        else:
            self.model_config = raw_config
            if not use_v4:
                use_v4 = any(
                    key in self.model_config
                    for key in ("supcon_dim", "dino_use_cls", "cnn_cross_view_attn")
                )

        self.image_size = int(self.model_config.get("image_size", 336))

        model_cls = _OralHealthModelV4 if use_v4 else _OralHealthModel

        model_paths = ec.get("models", [])
        loaded = 0
        for mp in model_paths:
            mp_path = Path(mp)
            if not mp_path.is_absolute():
                mp_path = (ensemble_path.parent / mp_path).resolve()
            if not mp_path.exists():
                logger.warning("Checkpoint missing: %s", mp)
                continue
            try:
                model = model_cls(self.model_config).to(self.device)
                ckpt = torch.load(str(mp_path), map_location=self.device, weights_only=False)

                # Support both full checkpoint dicts and raw state dicts
                if "model_state_dict" in ckpt:
                    state = ckpt["model_state_dict"]
                else:
                    state = ckpt

                if any(k.startswith("module.") for k in state):
                    state = {
                        (k[7:] if k.startswith("module.") else k): v
                        for k, v in state.items()
                    }

                missing, unexpected = model.load_state_dict(state, strict=False)
                loaded_params = len(model.state_dict()) - len(missing)
                if loaded_params == 0:
                    raise RuntimeError("checkpoint incompatible with configured architecture")

                if missing:
                    logger.info("Model loaded with %d missing keys", len(missing))
                if unexpected:
                    logger.info("Model loaded with %d unexpected keys", len(unexpected))

                model.eval()
                self.models.append(model)
                loaded += 1
                logger.info("Loaded fold model from %s", mp_path)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", mp_path, exc)

        logger.info("Loaded %d / %d ensemble models.", loaded, len(model_paths))

    # -----------------------------------------------------------------------
    # PI calibration
    # -----------------------------------------------------------------------

    def _load_pi_calibration(self, pi_calibration_path: str) -> None:
        self.pi_calibration: Dict = {}
        if os.path.exists(pi_calibration_path):
            with open(pi_calibration_path) as f:
                self.pi_calibration = json.load(f)
            logger.info("PI calibration loaded from %s", pi_calibration_path)
        else:
            logger.warning("PI calibration not found at %s — using defaults.", pi_calibration_path)

    # -----------------------------------------------------------------------
    # Warmup
    # -----------------------------------------------------------------------

    def _warmup(self) -> None:
        """Run a dummy forward pass to JIT compile kernels."""
        if not self.models:
            return
        dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
        with torch.no_grad():
            for m in self.models:
                try:
                    m(dummy, dummy, dummy)
                except Exception:
                    pass
        logger.info("Model warmup complete.")

    # -----------------------------------------------------------------------
    # Image → tensor
    # -----------------------------------------------------------------------

    def _image_to_tensor(self, image_path: str) -> torch.Tensor:
        """Load and preprocess one image to a model-ready tensor.

        Args:
            image_path: Path to the image.

        Returns:
            Float tensor of shape (1, C, H, W) on self.device.
        """
        img_rgb = _standardize_image(image_path, self.image_size)
        transformed = self.val_transform(image=img_rgb)
        return transformed["image"].unsqueeze(0).to(self.device)

    # -----------------------------------------------------------------------
    # DL prediction: MGI / OHI / GEI
    # -----------------------------------------------------------------------

    def _predict_dl(
        self,
        frontal_tensor: torch.Tensor,
        left_tensor: torch.Tensor,
        right_tensor: torch.Tensor,
    ) -> Dict[str, Dict[str, Any]]:
        """Run ensemble inference for a 3-view triplet.

        Args:
            frontal_tensor, left_tensor, right_tensor: (1, C, H, W) tensors.

        Returns:
            Dict: task → score + averaged class probabilities.
        """
        # NEW multi-view models
        if self.models:
            all_cls_probs: Dict[str, List[np.ndarray]] = {"mgi": [], "ohi": [], "gei": []}
            all_ord_probs: Dict[str, List[np.ndarray]] = {"mgi": [], "ohi": [], "gei": []}
            with torch.no_grad():
                for model in self.models:
                    results = model.predict_scores(
                        frontal_tensor,
                        left_tensor,
                        right_tensor,
                        thresholds=self.decode_thresholds,
                        temperature=self.decode_temperature,
                    )
                    for task in ("mgi", "ohi", "gei"):
                        all_cls_probs[task].append(results[task]["probs"].cpu().numpy()[0])
                        all_ord_probs[task].append(results[task]["ordinal_probs"].cpu().numpy()[0])

            out: Dict[str, Dict[str, Any]] = {}
            for task in ("mgi", "ohi", "gei"):
                p_cls = np.mean(all_cls_probs[task], axis=0)
                p_ord = np.mean(all_ord_probs[task], axis=0)
                thr = self.decode_thresholds.get(task, 0.5)
                score = _decode_ordinal_probs_np(p_ord, thr, task=task)
                out[task] = {
                    "score": score,
                    "probabilities": p_cls,
                }
            return out

        # No model available — return uniform distribution
        logger.error("No models available for inference. Returning uniform probabilities.")
        n_mgi = int(self.model_config.get("num_classes_mgi", 5))
        n_ohi = int(self.model_config.get("num_classes_ohi", 4))
        n_gei = int(self.model_config.get("num_classes_gei", 4))
        return {
            "mgi": {"score": 0, "probabilities": np.ones(n_mgi) / max(n_mgi, 1)},
            "ohi": {"score": 0, "probabilities": np.ones(n_ohi) / max(n_ohi, 1)},
            "gei": {"score": 0, "probabilities": np.ones(n_gei) / max(n_gei, 1)},
        }

        # Deprecated: PI computation is handled by pi_estimator.py via views.py
        pass

    # -----------------------------------------------------------------------
    # Public predict API
    # -----------------------------------------------------------------------

    def predict(
        self,
        frontal_path: str,
        left_path: str,
        right_path: str,
    ) -> Dict[str, Any]:
        """Predict all dental indices from three intraoral photographs.

        Args:
            frontal_path: Path to frontal view image.
            left_path: Path to left lateral view image.
            right_path: Path to right lateral view image.

        Returns:
            Dict with keys mgi, ohi, gei (score/confidence/label/probabilities)
            and pi (score/ratio/label/per_view/variability/confidence).

        Raises:
            ValueError: If any image file does not exist.
        """
        for label, path in [("frontal", frontal_path), ("left", left_path), ("right", right_path)]:
            if not os.path.exists(path):
                raise ValueError(f"{label} image not found: {path}")

        t0 = time.perf_counter()

        # ── Load and preprocess images ──────────────────────────────────
        frontal_t = self._image_to_tensor(frontal_path)
        left_t    = self._image_to_tensor(left_path)
        right_t   = self._image_to_tensor(right_path)

        # ── DL predictions ─────────────────────────────────────────────
        dl_pred = self._predict_dl(frontal_t, left_t, right_t)

        dl_results: Dict[str, Any] = {}
        for task in ("mgi", "ohi", "gei"):
            p = np.asarray(dl_pred[task]["probabilities"], dtype=np.float64)
            if p.size == 0:
                p = np.array([1.0], dtype=np.float64)

            score = int(dl_pred[task]["score"])
            score = int(np.clip(score, 0, p.size - 1))
            confidence = float(p[score])
            dl_results[task] = {
                "score": score,
                "confidence": confidence,
                "label": _TASK_LABELS[task].get(score, f"Class {score}"),
                "probabilities": p.tolist(),
                "low_confidence": confidence < 0.50,
            }
            logger.info(
                "%s → score=%d conf=%.1f%% label=%s",
                task.upper(), score, confidence * 100, dl_results[task]["label"]
            )

        # ── PI computation ─────────────────────────────────────────────
        # PI calculation is now handled explicitly in views.py
        # using the computer-vision based inference/pi_estimator.py.
        pi_final = {}

        self._last_latency = time.perf_counter() - t0
        logger.info("Predict completed in %.3fs", self._last_latency)

        return {
            "mgi": dl_results["mgi"],
            "ohi": dl_results["ohi"],
            "gei": dl_results["gei"],
            "pi": pi_final,
            "low_confidence_warning": any(
                dl_results[t]["low_confidence"] for t in ("mgi", "ohi", "gei")
            ),
        }

    @property
    def last_inference_latency(self) -> float:
        return self._last_latency


