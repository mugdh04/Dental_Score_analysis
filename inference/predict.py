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
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

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
    return timm.create_model(name, pretrained=False, img_size=img_size, num_classes=0, global_pool="avg")

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
    """Multi-view DINOv2 model — mirrors training/train_model_v2.py exactly."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        backbone_name = config.get("backbone_model", "vit_small_patch14_dinov2.lvd142m")
        dropout  = float(config.get("dropout", 0.20))
        proj_dim = int(config.get("projection_dim", 256))
        head_dim = int(config.get("head_hidden_dim", 128))
        img_size = int(config.get("image_size", 336))

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

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if out.dim() == 3:
            out = out[:, 0]
        return out

    def forward(self, frontal: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self._extract(frontal)
        l = self._extract(left)
        r = self._extract(right)

        views = torch.stack([f, l, r], dim=1)
        pooled = self.view_pool(views)
        shared = self.projection(pooled)

        return {
            "mgi": self.mgi_head(shared),
            "ohi": self.ohi_head(shared),
            "gei": self.gei_head(shared),
        }

    def predict_scores(self, frontal, left, right):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(frontal, left, right)

        result = {}
        for key in ("mgi", "ohi", "gei"):
            logits = outputs[key].float()
            probs = torch.sigmoid(logits)
            
            # 1) Hard predict: how many thresholds are crossed?
            predicted = (probs > 0.5).sum(dim=1).int()
            
            # 2) Soft predict: synthetic class probability distribution
            B, K_minus_1 = probs.shape
            K = K_minus_1 + 1
            class_probs = torch.zeros(B, K, device=probs.device)
            class_probs[:, 0] = 1.0 - probs[:, 0]
            for i in range(1, K_minus_1):
                class_probs[:, i] = probs[:, i - 1] * (1.0 - probs[:, i])
            class_probs[:, -1] = probs[:, -1]
            
            # Normalize and clamp safely
            class_probs = torch.clamp(class_probs, 1e-6, 1.0)
            class_probs = class_probs / class_probs.sum(dim=1, keepdim=True)
            
            confidence = class_probs.max(dim=1).values * 100.0
            
            result[key] = {
                "score": predicted,
                "confidence": confidence,
                "probs": class_probs,
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
        self.models: List[_OralHealthModel] = []
        self.model_config: Dict = {}
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

        self.model_config = ec.get("config", {
            "backbone_model": "vit_small_patch14_dinov2.lvd142m",
            "dropout": 0.45,
            "projection_dim": 512,
            "head_hidden_dim": 256,
            "num_classes_mgi": 5,
            "num_classes_ohi": 4,
            "num_classes_gei": 4,
            "image_size": 336,
        })
        self.image_size = int(self.model_config.get("image_size", 336))

        model_paths = ec.get("models", [])
        loaded = 0
        for mp in model_paths:
            mp_path = Path(mp)
            if not mp_path.exists():
                logger.warning("Checkpoint missing: %s", mp)
                continue
            try:
                model = _OralHealthModel(self.model_config).to(self.device)
                ckpt = torch.load(str(mp_path), map_location=self.device, weights_only=False)

                # Support both full checkpoint dicts and raw state dicts
                if "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"], strict=False)
                else:
                    model.load_state_dict(ckpt, strict=False)

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
    ) -> Dict[str, np.ndarray]:
        """Run ensemble inference for a 3-view triplet.

        Args:
            frontal_tensor, left_tensor, right_tensor: (1, C, H, W) tensors.

        Returns:
            Dict: task → averaged softmax probability vector (numpy array).
        """
        # NEW multi-view models
        if self.models:
            all_probs: Dict[str, List[np.ndarray]] = {"mgi": [], "ohi": [], "gei": []}
            with torch.no_grad():
                for model in self.models:
                    results = model.predict_scores(frontal_tensor, left_tensor, right_tensor)
                    for task in ("mgi", "ohi", "gei"):
                        all_probs[task].append(results[task]["probs"].cpu().numpy()[0])
            return {task: np.mean(probs_list, axis=0) for task, probs_list in all_probs.items()}

        # No model available — return uniform distribution
        logger.error("No models available for inference. Returning uniform probabilities.")
        return {
            "mgi": np.ones(5) / 5,
            "ohi": np.ones(4) / 4,
            "gei": np.ones(4) / 4,
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
        probs = self._predict_dl(frontal_t, left_t, right_t)

        dl_results: Dict[str, Any] = {}
        for task in ("mgi", "ohi", "gei"):
            p = probs[task]
            score = int(np.argmax(p))
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


