"""SAM-based gum and tooth mask generation with safe fallbacks."""

from __future__ import annotations

import logging
import importlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GumSegmentor:
    """Generate tooth and gum masks from box prompts using Segment Anything.

    The class supports graceful degradation when SAM is unavailable or no
    checkpoint is provided by approximating masks from bounding boxes.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "vit_b",
        device: str = "cpu",
        border_dilation_px: int = 15,
    ) -> None:
        """Initialize the segmentor.

        Args:
            checkpoint_path: Path to SAM checkpoint. If absent, fallback mode is used.
            model_type: SAM model type key (e.g., "vit_b").
            device: Device string for SAM inference.
            border_dilation_px: Border width in pixels for gum-region approximation.
        """
        self.border_dilation_px = border_dilation_px
        self._predictor = None

        if checkpoint_path is None or not Path(checkpoint_path).exists():
            if checkpoint_path:
                logger.warning("SAM checkpoint not found at %s. Using fallback masks.", checkpoint_path)
            else:
                logger.info("No SAM checkpoint provided. Using fallback masks.")
            return

        try:
            sam_module = importlib.import_module('segment_anything')
            SamPredictor = getattr(sam_module, 'SamPredictor')
            sam_model_registry = getattr(sam_module, 'sam_model_registry')

            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=device)
            self._predictor = SamPredictor(sam)
            logger.info("Loaded SAM model type=%s checkpoint=%s", model_type, checkpoint_path)
        except Exception as exc:
            logger.warning("SAM initialization failed. Using fallback masks. Error: %s", exc)
            self._predictor = None

    @staticmethod
    def _boxes_to_mask(image_shape: Tuple[int, int], boxes: Sequence[Sequence[float]]) -> np.ndarray:
        """Build a coarse binary mask by filling all bounding boxes."""
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            mask[y1 : y2 + 1, x1 : x2 + 1] = 1
        return mask

    def _gum_from_tooth(self, tooth_mask: np.ndarray) -> np.ndarray:
        """Approximate gum as a dilated border ring around tooth mask."""
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(tooth_mask.astype(np.uint8), kernel, iterations=max(1, self.border_dilation_px // 3))
        gum = np.clip(dilated - tooth_mask.astype(np.uint8), 0, 1)
        return gum

    def segment(self, image: np.ndarray, bounding_boxes: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Segment tooth and gum masks from an image and box prompts.

        Args:
            image: RGB image as HxWx3 numpy array.
            bounding_boxes: Iterable of [x1, y1, x2, y2, conf] style boxes.

        Returns:
            A tuple (tooth_mask, gum_mask), each as HxW uint8 mask values in {0,1}.
        """
        if image is None or image.ndim != 3:
            raise ValueError("GumSegmentor.segment expects an HxWx3 RGB image.")

        h, w = image.shape[:2]
        if not bounding_boxes:
            bounding_boxes = [[0.0, 0.0, float(w - 1), float(h - 1), 1.0]]

        fallback_tooth = self._boxes_to_mask((h, w), bounding_boxes)

        if self._predictor is None:
            return fallback_tooth, self._gum_from_tooth(fallback_tooth)

        try:
            image_uint8 = image if image.dtype == np.uint8 else (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
            self._predictor.set_image(image_uint8)

            combined = np.zeros((h, w), dtype=np.uint8)
            for box in bounding_boxes:
                x1, y1, x2, y2 = box[:4]
                input_box = np.array([x1, y1, x2, y2], dtype=np.float32)
                masks, scores, _ = self._predictor.predict(
                    box=input_box,
                    multimask_output=True,
                )

                if masks is None or len(masks) == 0:
                    continue

                best_idx = int(np.argmax(scores))
                combined = np.logical_or(combined, masks[best_idx]).astype(np.uint8)

            if combined.sum() == 0:
                logger.debug("SAM returned empty mask. Using bounding-box fallback mask.")
                combined = fallback_tooth

            gum_mask = self._gum_from_tooth(combined)
            return combined.astype(np.uint8), gum_mask.astype(np.uint8)
        except Exception as exc:
            logger.warning("SAM segmentation failed. Using fallback masks. Error: %s", exc)
            return fallback_tooth, self._gum_from_tooth(fallback_tooth)
