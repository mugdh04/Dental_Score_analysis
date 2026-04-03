"""Patch extraction pipeline for tooth and gum regions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .standardize import standardize_image

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


def _sanitize_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp a float box to integer image bounds."""
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


def _resize_patch(patch: np.ndarray, patch_size: int) -> np.ndarray:
    """Resize a patch to a square patch_size x patch_size."""
    return cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)


def generate_patches(
    image: np.ndarray,
    bounding_boxes: Sequence[Sequence[float]],
    masks: Tuple[np.ndarray, np.ndarray],
    patch_size: int,
) -> List[np.ndarray]:
    """Generate tooth and gum-border patches from detection boxes and masks.

    Args:
        image: RGB image array in [0,1] or uint8 format.
        bounding_boxes: Tooth bounding boxes in [x1,y1,x2,y2,confidence] format.
        masks: Tuple (tooth_mask, gum_mask), each HxW with values 0/1.
        patch_size: Patch output size.

    Returns:
        List of RGB uint8 patches.
    """
    if image is None or image.ndim != 3:
        raise ValueError("generate_patches expects an HxWx3 image.")

    image_uint8 = image if image.dtype == np.uint8 else (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    tooth_mask, gum_mask = masks

    h, w = image_uint8.shape[:2]
    patches: List[np.ndarray] = []

    for box in bounding_boxes:
        x1, y1, x2, y2 = _sanitize_box(box, width=w, height=h)
        if x2 <= x1 or y2 <= y1:
            continue

        tooth_crop = image_uint8[y1 : y2 + 1, x1 : x2 + 1]
        if tooth_crop.size == 0:
            continue

        patches.append(_resize_patch(tooth_crop, patch_size))

        # Gum-border patch for inflammation cues.
        gum_local = gum_mask[y1 : y2 + 1, x1 : x2 + 1]
        gum_crop = tooth_crop.copy()
        gum_crop[gum_local <= 0] = 0

        if int(np.count_nonzero(gum_local)) > 0:
            patches.append(_resize_patch(gum_crop, patch_size))

    if not patches:
        patches.append(_resize_patch(image_uint8, patch_size))

    return patches


def preprocess_all_images(
    dataset: Sequence[Dict[str, Any]],
    detector: Any,
    segmentor: Any,
    patch_dir: PathLike,
    image_size: int,
    patch_size: int,
) -> pd.DataFrame:
    """Run full preprocessing and save patch images + labels mapping CSV.

    Args:
        dataset: Iterable of image-level records containing patient_id, view,
            image_path, and labels mgi/ohi/gei.
        detector: ToothDetector-like instance exposing detect(image).
        segmentor: GumSegmentor-like instance exposing segment(image, boxes).
        patch_dir: Directory where patch images and labels CSV are written.
        image_size: Standardized image size before detection/segmentation.
        patch_size: Output patch size.

    Returns:
        DataFrame with patch-level paths and labels.
    """
    output_dir = Path(patch_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for entry in tqdm(dataset, desc="Preprocessing images"):
        patient_id = str(entry.get("patient_id", "unknown"))
        view = str(entry.get("view", "unknown"))
        image_path = entry.get("image_path")

        if image_path is None:
            logger.warning("Skipping entry with missing image_path: %s", entry)
            continue

        try:
            image = standardize_image(image_path=image_path, image_size=image_size)
            boxes = detector.detect(image)
            masks = segmentor.segment(image, boxes)
            patches = generate_patches(image=image, bounding_boxes=boxes, masks=masks, patch_size=patch_size)

            for idx, patch in enumerate(patches):
                filename = f"{patient_id}_{view}_{idx}.jpg"
                patch_path = output_dir / filename
                # cv2 expects BGR for writing
                patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                ok = cv2.imwrite(str(patch_path), patch_bgr)
                if not ok:
                    raise IOError(f"Failed to write patch image: {patch_path}")

                rows.append(
                    {
                        "patch_path": str(patch_path),
                        "patient_id": patient_id,
                        "view": view,
                        "mgi": int(entry["mgi"]),
                        "ohi": int(entry["ohi"]),
                        "gei": int(entry["gei"]),
                        "source_image": str(image_path),
                    }
                )
        except Exception as exc:
            logger.exception("Failed preprocessing for patient=%s view=%s path=%s: %s", patient_id, view, image_path, exc)
            continue

    patch_df = pd.DataFrame(rows)
    csv_path = output_dir / "patches_labels.csv"
    patch_df.to_csv(csv_path, index=False)
    logger.info("Saved %d patches metadata rows to %s", len(patch_df), csv_path)
    return patch_df
