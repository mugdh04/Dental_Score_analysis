"""YOLOv8-based tooth detection wrapper with robust fallbacks."""

from __future__ import annotations

import logging
import importlib
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class ToothDetector:
    """Detect tooth bounding boxes using a YOLOv8 model.

    The detector returns padded bounding boxes in the form
    [x1, y1, x2, y2, confidence]. If detection fails or no boxes are found,
    it falls back to the full image as one bounding box.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        weights_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        padding_ratio: float = 0.10,
    ) -> None:
        """Initialize the detector.

        Args:
            model_name: Default YOLOv8 model name when custom weights are not provided.
            weights_path: Optional path to custom/fine-tuned YOLO weights.
            conf_threshold: Detection confidence threshold.
            iou_threshold: NMS IoU threshold.
            device: Inference device string for Ultralytics.
            padding_ratio: Relative padding applied around each bounding box.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.padding_ratio = padding_ratio
        self._model = None

        try:
            ultralytics_module = importlib.import_module('ultralytics')
            YOLO = getattr(ultralytics_module, 'YOLO')

            selected = weights_path if weights_path else model_name
            if weights_path and not Path(weights_path).exists():
                logger.warning("Custom YOLO weights not found at %s. Falling back to %s.", weights_path, model_name)
                selected = model_name

            self._model = YOLO(selected)
            logger.info("Loaded YOLO model: %s", selected)
        except Exception as exc:
            logger.warning("YOLO initialization failed. Detector will use full-image fallback. Error: %s", exc)
            self._model = None

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        """Convert image to uint8 RGB for model input."""
        if image.dtype == np.uint8:
            return image
        clipped = np.clip(image, 0.0, 1.0)
        return (clipped * 255.0).astype(np.uint8)

    @staticmethod
    def _full_image_box(image: np.ndarray) -> List[List[float]]:
        """Return fallback full-image box with confidence 1.0."""
        h, w = image.shape[:2]
        return [[0.0, 0.0, float(max(w - 1, 0)), float(max(h - 1, 0)), 1.0]]

    def _pad_box(self, box: Sequence[float], width: int, height: int) -> List[float]:
        """Apply symmetric padding around a detected box while clipping to image bounds."""
        x1, y1, x2, y2, conf = box
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)

        pad_x = box_w * self.padding_ratio
        pad_y = box_h * self.padding_ratio

        px1 = max(0.0, x1 - pad_x)
        py1 = max(0.0, y1 - pad_y)
        px2 = min(float(width - 1), x2 + pad_x)
        py2 = min(float(height - 1), y2 + pad_y)
        return [px1, py1, px2, py2, conf]

    def detect(self, image: np.ndarray) -> List[List[float]]:
        """Detect tooth regions in an image.

        Args:
            image: RGB image as HxWx3 numpy array.

        Returns:
            List of bounding boxes [x1, y1, x2, y2, confidence].
        """
        if image is None or image.ndim != 3:
            raise ValueError("ToothDetector.detect expects an HxWx3 image array.")

        h, w = image.shape[:2]
        if self._model is None:
            return self._full_image_box(image)

        try:
            image_uint8 = self._to_uint8(image)
            results = self._model.predict(
                source=image_uint8,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

            boxes: List[List[float]] = []
            if results and len(results) > 0 and results[0].boxes is not None:
                for xyxy, conf in zip(results[0].boxes.xyxy, results[0].boxes.conf):
                    x1, y1, x2, y2 = xyxy.tolist()
                    boxes.append([float(x1), float(y1), float(x2), float(y2), float(conf.item())])

            if not boxes:
                logger.debug("No YOLO detections. Using full-image fallback.")
                return self._full_image_box(image)

            padded = [self._pad_box(box, width=w, height=h) for box in boxes]
            return padded
        except Exception as exc:
            logger.warning("YOLO detection failed. Using full-image fallback. Error: %s", exc)
            return self._full_image_box(image)
