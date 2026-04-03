"""Inference module with patch-based multi-task prediction and plaque scoring."""

from __future__ import annotations

import logging
import importlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from inference.plaque_index import compute_plaque_index, plaque_score_to_label
from preprocessing.patch_generator import generate_patches
from preprocessing.sam_segmentation import GumSegmentor
from preprocessing.standardize import standardize_image
from preprocessing.yolo_detection import ToothDetector
from training.model import OralHealthModel

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG: Dict[str, Any] = {
    "image_size": 512,
    "patch_size": 224,
    "dropout": 0.4,
    "num_classes_mgi": 5,
    "num_classes_ohi": 4,
    "num_classes_gei": 3,
    "backbone_model": "vit_small_patch14_dinov2.lvd142m",
    "yolo_model": "yolov8n.pt",
    "yolo_weights": None,
    "sam_checkpoint": None,
    "sam_model_type": "vit_b",
}


class OralHealthPredictor:
    """Patch-based oral health predictor for MGI/OHI/GEI and plaque index.

    The predictor standardizes input views, detects teeth, segments gum/tooth
    regions, generates patches, predicts class probabilities per patch, and
    aggregates view-level and case-level results.
    """

    def __init__(self, model_path: str, device: str = "cpu", config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize model and preprocessing components.

        Args:
            model_path: Path to trained model checkpoint.
            device: Inference device string.
            config: Optional inference config overrides.
        """
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        self.config = dict(_DEFAULT_CONFIG)
        if config:
            self.config.update(config)

        # Load checkpoint and merge embedded config if available.
        checkpoint = torch.load(str(self.model_path), map_location=self.device)
        checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        if isinstance(checkpoint_config, dict):
            self.config.update(checkpoint_config)

        self.model = OralHealthModel(config=self.config).to(self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.detector = ToothDetector(
            model_name=str(self.config.get("yolo_model", "yolov8n.pt")),
            weights_path=self.config.get("yolo_weights"),
            device=str(device),
        )
        self.segmentor = GumSegmentor(
            checkpoint_path=self.config.get("sam_checkpoint"),
            model_type=str(self.config.get("sam_model_type", "vit_b")),
            device=str(device),
        )

        try:
            A = importlib.import_module("albumentations")
            to_tensor_module = importlib.import_module("albumentations.pytorch")
            ToTensorV2 = getattr(to_tensor_module, "ToTensorV2")
        except Exception as exc:
            raise ImportError(
                "Albumentations is required for OralHealthPredictor. "
                "Install model dependencies from requirements_model.txt."
            ) from exc

        self.val_transform = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def _predict_patch_batch(self, patches: Sequence[np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Predict per-task class outputs from a list of RGB patches."""
        if not patches:
            raise ValueError("No patches provided to _predict_patch_batch.")

        tensors: List[torch.Tensor] = []
        for patch in patches:
            transformed = self.val_transform(image=patch)
            tensors.append(transformed["image"])

        batch = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)

        output: Dict[str, Dict[str, float]] = {}
        for task in ("mgi", "ohi", "gei"):
            probs = torch.softmax(logits[task], dim=1)
            mean_probs = probs.mean(dim=0)
            pred_class = int(torch.argmax(mean_probs).item())
            confidence = float(mean_probs[pred_class].item())
            output[task] = {
                "score": pred_class,
                "confidence": confidence,
            }
        return output

    def _prepare_view(self, image_path: str) -> Tuple[List[np.ndarray], List[float]]:
        """Run standardize/detect/segment/patch and compute plaque ratios for one view."""
        image = standardize_image(image_path=image_path, image_size=int(self.config["image_size"]))
        boxes = self.detector.detect(image)
        tooth_mask, gum_mask = self.segmentor.segment(image=image, bounding_boxes=boxes)
        patches = generate_patches(
            image=image,
            bounding_boxes=boxes,
            masks=(tooth_mask, gum_mask),
            patch_size=int(self.config["patch_size"]),
        )

        image_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        plaque_ratios: List[float] = []
        for box in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
            x1 = max(0, min(image_uint8.shape[1] - 1, x1))
            y1 = max(0, min(image_uint8.shape[0] - 1, y1))
            x2 = max(0, min(image_uint8.shape[1] - 1, x2))
            y2 = max(0, min(image_uint8.shape[0] - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            crop_image = image_uint8[y1 : y2 + 1, x1 : x2 + 1]
            crop_mask = tooth_mask[y1 : y2 + 1, x1 : x2 + 1]
            ratio, _ = compute_plaque_index(crop_image, crop_mask)
            plaque_ratios.append(float(ratio))

        if not plaque_ratios:
            # Fallback to full image if detection-level plaque ratio was unavailable.
            ratio, _ = compute_plaque_index(image_uint8, tooth_mask)
            plaque_ratios.append(float(ratio))

        return patches, plaque_ratios

    @staticmethod
    def _majority_vote(scores: Sequence[int], confidences: Sequence[float]) -> Tuple[int, float]:
        """Majority vote with confidence-based tie-break."""
        counts = Counter(scores)
        max_count = max(counts.values())
        tied = [cls for cls, count in counts.items() if count == max_count]

        if len(tied) == 1:
            winner = tied[0]
        else:
            best_conf = -1.0
            winner = tied[0]
            for cls in tied:
                cls_conf = [conf for score, conf in zip(scores, confidences) if score == cls]
                mean_conf = float(np.mean(cls_conf)) if cls_conf else 0.0
                if mean_conf > best_conf:
                    best_conf = mean_conf
                    winner = cls

        winner_confs = [conf for score, conf in zip(scores, confidences) if score == winner]
        winner_conf = float(np.mean(winner_confs)) if winner_confs else float(np.max(confidences))
        return int(winner), winner_conf

    def predict(self, frontal_path: str, left_path: str, right_path: str) -> Dict[str, Any]:
        """Predict MGI/OHI/GEI and plaque index from three intraoral views.

        Args:
            frontal_path: Path to frontal-view image.
            left_path: Path to left-lateral image.
            right_path: Path to right-lateral image.

        Returns:
            Dictionary with scores and confidences for MGI/OHI/GEI, and PI ratio/score/label.
        """
        view_paths = {
            "frontal": frontal_path,
            "left": left_path,
            "right": right_path,
        }

        per_view: Dict[str, Dict[str, Dict[str, float]]] = {}
        all_plaque_ratios: List[float] = []

        for view_name, path in view_paths.items():
            patches, plaque_ratios = self._prepare_view(path)
            per_view[view_name] = self._predict_patch_batch(patches)
            all_plaque_ratios.extend(plaque_ratios)

        aggregated: Dict[str, Dict[str, float]] = {}
        for task in ("mgi", "ohi", "gei"):
            scores = [int(per_view[v][task]["score"]) for v in ("frontal", "left", "right")]
            confs = [float(per_view[v][task]["confidence"]) for v in ("frontal", "left", "right")]
            score, conf = self._majority_vote(scores, confs)
            aggregated[task] = {"score": score, "confidence": conf}

        pi_ratio = float(np.mean(all_plaque_ratios)) if all_plaque_ratios else 0.0
        if pi_ratio <= 0.10:
            pi_score = 0
        elif pi_ratio <= 0.25:
            pi_score = 1
        elif pi_ratio <= 0.50:
            pi_score = 2
        else:
            pi_score = 3

        return {
            "mgi": aggregated["mgi"],
            "ohi": aggregated["ohi"],
            "gei": aggregated["gei"],
            "pi": {
                "ratio": pi_ratio,
                "score": pi_score,
                "label": plaque_score_to_label(pi_score),
            },
        }
