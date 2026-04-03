"""Rule-based plaque index computation from tooth image regions."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def plaque_score_to_label(score: int) -> str:
    """Map plaque score to categorical label."""
    return {
        0: "None",
        1: "Low",
        2: "Medium",
        3: "High",
    }.get(int(score), "Unknown")


def compute_plaque_index(tooth_image: np.ndarray, tooth_mask: np.ndarray) -> Tuple[float, int]:
    """Compute plaque ratio and categorical score from a tooth region.

    Args:
        tooth_image: RGB uint8 or float image for a tooth region.
        tooth_mask: Binary mask (same HxW) indicating tooth pixels.

    Returns:
        Tuple of (plaque_ratio, plaque_score).

    Notes:
        - HSV mask captures yellow-brown plaque tones.
        - LAB-b channel threshold captures additional yellowing.
        - Morphology removes noise before computing ratio.
    """
    if tooth_image is None or tooth_image.ndim != 3:
        return 0.0, 0

    if tooth_mask is None or tooth_mask.ndim != 2:
        tooth_mask = np.ones(tooth_image.shape[:2], dtype=np.uint8)

    image_uint8 = tooth_image if tooth_image.dtype == np.uint8 else (np.clip(tooth_image, 0.0, 1.0) * 255.0).astype(np.uint8)
    mask_bin = (tooth_mask > 0).astype(np.uint8)

    total_tooth_pixels = int(mask_bin.sum())
    if total_tooth_pixels == 0:
        return 0.0, 0

    hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    hsv_mask = (
        (h >= 15)
        & (h <= 40)
        & (s > 50)
        & (v >= 80)
        & (v <= 220)
    )

    lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
    _, _, b_chan = cv2.split(lab)
    lab_mask = b_chan > 140

    combined = (hsv_mask | lab_mask).astype(np.uint8)
    combined = combined * mask_bin

    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.erode(combined, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=2)

    plaque_pixels = int(cleaned.sum())
    plaque_ratio = float(plaque_pixels) / float(total_tooth_pixels)

    if plaque_ratio <= 0.10:
        score = 0
    elif plaque_ratio <= 0.25:
        score = 1
    elif plaque_ratio <= 0.50:
        score = 2
    else:
        score = 3

    return plaque_ratio, score
