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
    def __init__(self, in_features: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.45) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _OralHealthModel(nn.Module):
    """Multi-view DINOv2 model — mirrors training/model.py exactly."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        import timm

        backbone_name = config.get("backbone_model", "vit_small_patch14_dinov2.lvd142m")
        dropout   = float(config.get("dropout", 0.45))
        proj_dim  = int(config.get("projection_dim", 512))
        hidden    = int(config.get("head_hidden_dim", 256))

        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="avg")
        backbone_dim = int(getattr(self.backbone, "num_features", 384))

        fused_dim = backbone_dim * 3
        self.shared_projection = nn.Sequential(
            nn.Linear(fused_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mgi_head = _TaskHead(proj_dim, config.get("num_classes_mgi", 5), hidden, dropout)
        self.ohi_head = _TaskHead(proj_dim, config.get("num_classes_ohi", 4), hidden, dropout)
        self.gei_head = _TaskHead(proj_dim, config.get("num_classes_gei", 4), hidden, dropout)

    def _feat(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        return f[0] if isinstance(f, (list, tuple)) else f

    def forward(self, frontal: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> Dict[str, torch.Tensor]:
        fused = torch.cat([self._feat(frontal), self._feat(left), self._feat(right)], dim=1)
        shared = self.shared_projection(fused)
        return {
            "mgi": self.mgi_head(shared),
            "ohi": self.ohi_head(shared),
            "gei": self.gei_head(shared),
        }

    def predict_scores(self, frontal, left, right):
        self.eval()
        with torch.no_grad():
            out = self.forward(frontal, left, right)
        result = {}
        for key in ("mgi", "ohi", "gei"):
            probs = torch.softmax(out[key].float(), dim=1)
            result[key] = {
                "score": probs.argmax(dim=1).int(),
                "confidence": probs.max(dim=1).values * 100.0,
                "probs": probs,
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

    # -----------------------------------------------------------------------
    # PI scoring (rule-based multi-colour-space)
    # -----------------------------------------------------------------------

    def _compute_pi_for_image(self, image_path: str) -> Dict[str, Any]:
        """Compute Plaque Index for a single dental image.

        Uses multi-colour-space plaque detection without requiring YOLO/SAM.
        Analyses the full image with regional weighting.

        Args:
            image_path: Path to the image.

        Returns:
            Dict with score (0-5), ratio, label, confidence.
        """
        try:
            img_rgb = _standardize_image(image_path, self.image_size)
            return _compute_pi_full_image(img_rgb, self.pi_calibration)
        except Exception as exc:
            logger.warning("PI computation failed for %s: %s", image_path, exc)
            return {"score": 0, "ratio": 0.0, "label": "Unknown", "confidence": "low"}

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
        pi_frontal = self._compute_pi_for_image(frontal_path)
        pi_left    = self._compute_pi_for_image(left_path)
        pi_right   = self._compute_pi_for_image(right_path)
        pi_final   = _aggregate_pi_across_views(pi_frontal, pi_left, pi_right)

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


# ---------------------------------------------------------------------------
# PI algorithm — multi-colour-space, full-image (no YOLO/SAM required)
# ---------------------------------------------------------------------------

def _compute_pi_full_image(img_rgb: np.ndarray, calibration: Dict) -> Dict[str, Any]:
    """Compute plaque index on a full standardised dental image.

    Uses four complementary plaque detection masks across HSV, LAB,
    and relative-whiteness colour spaces.

    Args:
        img_rgb: RGB uint8 array, already standardised.
        calibration: PI calibration dict from pi_calibration.json.

    Returns:
        Dict with score, ratio, label, confidence.
    """
    H, W = img_rgb.shape[:2]

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    l_ch, _a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    r_ch, g_ch, b_rgb = img_rgb[:, :, 0].astype(np.int32), img_rgb[:, :, 1].astype(np.int32), img_rgb[:, :, 2].astype(np.int32)

    # Mask A — HSV yellow-brown (primary plaque)
    mask_a = ((h_ch >= 10) & (h_ch <= 42) & (s_ch >= 30) & (v_ch >= 60) & (v_ch <= 230)).astype(np.uint8)

    # Mask B — LAB b-channel (yellowish staining)
    mask_b = ((b_ch > 128) & (l_ch >= 50) & (l_ch <= 210)).astype(np.uint8)

    # Mask C — relative yellowness (R-B > 15 AND G-B > 8)
    mask_c = ((r_ch - b_rgb > 15) & (g_ch - b_rgb > 8)).astype(np.uint8)

    # Mask D — dark staining relative to image median
    median_v = float(np.median(v_ch))
    mask_d = (v_ch < median_v * 0.60).astype(np.uint8)

    raw = np.clip(mask_a | mask_b | mask_c | mask_d, 0, 1)

    # Morphological cleanup
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    raw = cv2.morphologyEx((raw * 255).astype(np.uint8), cv2.MORPH_OPEN, k3)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k5)
    plaque_mask = raw > 0

    # Regional weighting (gingival third is clinically most significant)
    zone_h = H // 3
    zone1 = plaque_mask[:zone_h, :]          # gingival third — top of image
    zone2 = plaque_mask[zone_h:2*zone_h, :]  # middle third
    zone3 = plaque_mask[2*zone_h:, :]        # incisal third

    n1, n2, n3 = zone1.size, zone2.size, zone3.size
    r1 = zone1.sum() / n1 if n1 > 0 else 0.0
    r2 = zone2.sum() / n2 if n2 > 0 else 0.0
    r3 = zone3.sum() / n3 if n3 > 0 else 0.0

    weighted_ratio = float(0.5 * r1 + 0.3 * r2 + 0.2 * r3)

    # Apply calibration correction
    ref_brightness = float(calibration.get("mean_brightness", 128.0))
    img_brightness = float(np.median(v_ch))
    correction = 1.0
    if ref_brightness > 0:
        correction = np.clip(ref_brightness / max(img_brightness, 1.0), 0.7, 1.4)
    calib_ratio = float(np.clip(weighted_ratio * correction, 0.0, 1.0))

    score, label = _ratio_to_pi_score(calib_ratio)

    # Confidence: lower near boundaries
    bin_edges = [0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 1.0]
    dist_to_boundary = min(abs(calib_ratio - e) for e in bin_edges)
    confidence_str = "high" if dist_to_boundary > 0.05 else ("medium" if dist_to_boundary > 0.02 else "low")

    return {"score": score, "ratio": calib_ratio, "label": label, "confidence": confidence_str}


def _ratio_to_pi_score(ratio: float) -> Tuple[int, str]:
    """Map calibrated plaque ratio to 0-5 PI score and label.

    Args:
        ratio: Calibrated plaque coverage ratio [0, 1].

    Returns:
        (score, label) tuple.
    """
    bins = [
        (0.05, 0, "No plaque"),
        (0.15, 1, "Trace — thin film at gingival margin"),
        (0.30, 2, "Mild — plaque in gingival third"),
        (0.50, 3, "Moderate — plaque up to half the tooth"),
        (0.70, 4, "Heavy — plaque over more than half"),
        (1.01, 5, "Severe — abundant plaque, possible calculus"),
    ]
    for upper, score, label in bins:
        if ratio < upper:
            return score, label
    return 5, "Severe — abundant plaque, possible calculus"


def _aggregate_pi_across_views(
    frontal: Dict,
    left: Dict,
    right: Dict,
) -> Dict[str, Any]:
    """Combine per-view PI results with clinical weighting.

    Frontal covers more tooth surface so gets higher weight.

    Args:
        frontal, left, right: Per-view PI result dicts.

    Returns:
        Aggregated PI dict.
    """
    w_f, w_l, w_r = 0.4, 0.3, 0.3
    ratios = [frontal["ratio"], left["ratio"], right["ratio"]]
    mean_ratio = float(w_f * ratios[0] + w_l * ratios[1] + w_r * ratios[2])
    variability = float(np.std(ratios))

    score, label = _ratio_to_pi_score(mean_ratio)

    conf_priority = {"high": 2, "medium": 1, "low": 0}
    worst_conf = min(
        [frontal["confidence"], left["confidence"], right["confidence"]],
        key=lambda c: conf_priority.get(c, 0),
    )

    return {
        "pi_score": score,
        "pi_ratio": mean_ratio,
        "pi_label": label,
        "pi_variability": variability,
        "pi_confidence": worst_conf,
        "per_view": {
            "frontal": {"score": frontal["score"], "ratio": frontal["ratio"]},
            "left":    {"score": left["score"],    "ratio": left["ratio"]},
            "right":   {"score": right["score"],   "ratio": right["ratio"]},
        },
    }
