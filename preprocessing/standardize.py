"""Image standardization helpers for dental photographs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


def standardize_image_array(image_rgb: np.ndarray, image_size: int) -> np.ndarray:
    """Standardize an in-memory RGB image using resize + CLAHE + normalization.

    Args:
        image_rgb: Input image in RGB format as an HxWx3 numpy array.
        image_size: Output square size in pixels.

    Returns:
        Standardized float32 image in RGB format with values in [0, 1].

    Raises:
        ValueError: If image is invalid or not a 3-channel array.
    """
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape HxWx3.")

    try:
        resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_chan = clahe.apply(l_chan)

        enhanced_lab = cv2.merge([l_chan, a_chan, b_chan])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        return enhanced_rgb.astype(np.float32) / 255.0
    except Exception as exc:
        raise ValueError(f"Failed to standardize image array: {exc}") from exc


def standardize_image(image_path: PathLike, image_size: int) -> np.ndarray:
    """Load and standardize an image from disk using CLAHE on the LAB L channel.

    Args:
        image_path: Path to the input image.
        image_size: Output square size in pixels.

    Returns:
        Standardized float32 image in RGB format with values in [0, 1].

    Raises:
        FileNotFoundError: If image path does not exist or cannot be read.
        ValueError: If standardization fails.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image file: {path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    standardized = standardize_image_array(image_rgb=image_rgb, image_size=image_size)
    logger.debug("Standardized image %s to shape %s", path, standardized.shape)
    return standardized
