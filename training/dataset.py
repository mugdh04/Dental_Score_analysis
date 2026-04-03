"""Dataset utilities for Thesis_Results.csv parsing and patch-level training."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


def _find_with_prefix(directory: Path, prefix: str) -> Optional[Path]:
    """Find an image path in directory by trying common image extensions."""
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = directory / f"{prefix}{ext}"
        if candidate.exists():
            return candidate
    return None


def _parse_scores_from_text(score_text: str) -> Tuple[int, int, int]:
    """Parse score text containing MGI/OHI/GEI numeric values.

    Args:
        score_text: Composite score string such as
            "MGI-0,OHI-1,GEI-0" or " MGI- 4, OHI-0,GEI-1".

    Returns:
        Tuple (mgi, ohi, gei).

    Raises:
        ValueError: If one of the score tokens cannot be extracted.
    """
    clean = str(score_text).strip()
    mgi_match = re.search(r"MGI\s*-\s*(\d+)", clean, flags=re.IGNORECASE)
    ohi_match = re.search(r"OHI\s*-\s*(\d+)", clean, flags=re.IGNORECASE)
    gei_match = re.search(r"GEI\s*-\s*(\d+)", clean, flags=re.IGNORECASE)

    if not (mgi_match and ohi_match and gei_match):
        raise ValueError(f"Could not parse score text: {score_text}")

    mgi = int(mgi_match.group(1))
    ohi = int(ohi_match.group(1))
    gei = int(gei_match.group(1))

    if not (0 <= mgi <= 4 and 0 <= ohi <= 3 and 0 <= gei <= 2):
        raise ValueError(f"Out-of-range score triple parsed from: {score_text}")

    return mgi, ohi, gei


def _load_from_thesis_results(data_dir: Path, labels_path: Path) -> List[Dict[str, Any]]:
    """Load image-level records using Thesis_Results.csv and F/L/R naming convention."""
    df = pd.read_csv(labels_path, usecols=[0, 1, 2], dtype=str)
    photo_root = data_dir / "Thesis_Photographs"

    view_map = {
        "frontal": (photo_root / "Frontal", "F"),
        "left": (photo_root / "Left_Lateral", "L"),
        "right": (photo_root / "Right_Lateral", "R"),
    }

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        sl_no_raw = row.iloc[0]
        name = str(row.iloc[1]).strip()
        score_text = row.iloc[2]

        try:
            patient_id = int(str(sl_no_raw).strip())
            mgi, ohi, gei = _parse_scores_from_text(str(score_text))
        except Exception as exc:
            logger.warning("Skipping malformed label row sl_no=%s name=%s reason=%s", sl_no_raw, name, exc)
            continue

        for view_name, (folder, prefix) in view_map.items():
            image_path = _find_with_prefix(folder, f"{prefix}{patient_id}")
            if image_path is None:
                logger.warning("Missing image for patient_id=%s view=%s", patient_id, view_name)
                continue

            records.append(
                {
                    "patient_id": str(patient_id),
                    "patient_name": name,
                    "view": view_name,
                    "image_path": str(image_path),
                    "mgi": mgi,
                    "ohi": ohi,
                    "gei": gei,
                    "label_source": str(labels_path),
                }
            )

    return records


def _load_from_labels_csv(data_dir: Path, labels_path: Path) -> List[Dict[str, Any]]:
    """Load image-level records from labels.csv format.

    Required columns: image_id, view, mgi, ohi, gei
    Optional columns: patient_id, image_path
    """
    df = pd.read_csv(labels_path)
    required = {"image_id", "view", "mgi", "ohi", "gei"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"labels.csv missing required columns: {sorted(missing)}")

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        image_id = str(row["image_id"]).strip()
        view = str(row["view"]).strip().lower()
        patient_id = str(row.get("patient_id", image_id.split("_")[0])).strip()

        if "image_path" in df.columns and pd.notna(row.get("image_path")):
            image_path = data_dir / str(row["image_path"]).strip()
        else:
            # Best-effort fallback: infer path as data_dir/view/image_id.*
            candidate_dir = data_dir / view
            image_path = _find_with_prefix(candidate_dir, image_id)

        if image_path is None or not Path(image_path).exists():
            logger.warning("Skipping labels.csv row with unresolved image path: image_id=%s", image_id)
            continue

        records.append(
            {
                "patient_id": patient_id,
                "patient_name": str(row.get("name", "")).strip(),
                "view": view,
                "image_path": str(image_path),
                "mgi": int(row["mgi"]),
                "ohi": int(row["ohi"]),
                "gei": int(row["gei"]),
                "label_source": str(labels_path),
            }
        )

    return records


def load_dataset(data_dir: PathLike) -> List[Dict[str, Any]]:
    """Load image-level records from either labels.csv or Thesis_Results.csv.

    Args:
        data_dir: Dataset directory path.

    Returns:
        List of dictionaries, one per image, containing image path, view, patient_id,
        and task labels.

    Raises:
        FileNotFoundError: If no supported labels file can be found.
    """
    root = Path(data_dir)
    candidates = [
        root / "labels.csv",
        root / "Thesis_Results.csv",
        root.parent / "Thesis_Results.csv",
    ]

    labels_path: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            labels_path = candidate
            break

    if labels_path is None:
        raise FileNotFoundError(
            f"No labels.csv or Thesis_Results.csv found under {root} or its parent."
        )

    if labels_path.name.lower() == "labels.csv":
        records = _load_from_labels_csv(root, labels_path)
    else:
        # If data_dir points directly to Thesis_Data this works as-is.
        # If data_dir points to another directory but has Thesis_Results in parent, use parent.
        thesis_root = labels_path.parent
        records = _load_from_thesis_results(thesis_root, labels_path)

    logger.info("Loaded %d image-level records from %s", len(records), labels_path)
    return records


class PatientDataset(Dataset):
    """Group image-level records by patient so three views are available together."""

    REQUIRED_VIEWS = ("frontal", "left", "right")

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        require_complete_views: bool = True,
        load_images: bool = False,
    ) -> None:
        """Initialize patient-level grouped dataset.

        Args:
            records: Image-level dictionaries from load_dataset.
            require_complete_views: If True, keep only patients with all 3 views.
            load_images: If True, __getitem__ returns loaded RGB arrays instead of paths.
        """
        self.load_images = load_images

        grouped: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"views": {}})
        for row in records:
            pid = str(row["patient_id"])
            grouped[pid]["patient_id"] = pid
            grouped[pid]["patient_name"] = row.get("patient_name", "")
            grouped[pid]["labels"] = {
                "mgi": int(row["mgi"]),
                "ohi": int(row["ohi"]),
                "gei": int(row["gei"]),
            }
            grouped[pid]["views"][str(row["view"]).lower()] = row["image_path"]

        samples: List[Dict[str, Any]] = []
        for sample in grouped.values():
            has_all = all(v in sample["views"] for v in self.REQUIRED_VIEWS)
            if require_complete_views and not has_all:
                continue
            sample["is_complete_triplet"] = has_all
            samples.append(sample)

        self.samples = sorted(samples, key=lambda x: int(x["patient_id"]) if str(x["patient_id"]).isdigit() else x["patient_id"])
        logger.info("Built patient-level dataset with %d samples", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        if not self.load_images:
            return sample

        loaded_views: Dict[str, np.ndarray] = {}
        for view_name, image_path in sample["views"].items():
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Failed to load image: {image_path}")
            loaded_views[view_name] = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        return {
            **sample,
            "views": loaded_views,
        }


class PatchDataset(Dataset):
    """Patch-level dataset reading from patches_labels.csv and applying augmentation."""

    def __init__(
        self,
        patch_df: Union[pd.DataFrame, PathLike],
        mode: str,
        train_transform: Optional[Any],
        val_transform: Optional[Any],
    ) -> None:
        """Initialize patch-level dataset.

        Args:
            patch_df: DataFrame or CSV path with patch metadata.
            mode: Either "train" or "val".
            train_transform: Albumentations train pipeline.
            val_transform: Albumentations val pipeline.
        """
        if mode not in {"train", "val"}:
            raise ValueError("PatchDataset mode must be 'train' or 'val'.")

        if isinstance(patch_df, (str, Path)):
            self.df = pd.read_csv(patch_df)
        else:
            self.df = patch_df.copy()

        self.df = self.df.reset_index(drop=True)
        self.transform = train_transform if mode == "train" else val_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        row = self.df.iloc[index]
        image_path = str(row["patch_path"])

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Patch image could not be loaded: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image_rgb)
            patch_tensor = transformed["image"]
        else:
            patch_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        targets = {
            "mgi": int(row["mgi"]),
            "ohi": int(row["ohi"]),
            "gei": int(row["gei"]),
        }
        return patch_tensor, targets


def get_dataloaders(
    fold_train_indices: Sequence[int],
    fold_val_indices: Sequence[int],
    patch_df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    train_transform: Optional[Any],
    val_transform: Optional[Any],
) -> Tuple[DataLoader, DataLoader]:
    """Build train/validation DataLoaders for one fold.

    Args:
        fold_train_indices: Row indices in patch_df for train split.
        fold_val_indices: Row indices in patch_df for validation split.
        patch_df: Patch metadata DataFrame.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        train_transform: Albumentations train transform.
        val_transform: Albumentations validation transform.

    Returns:
        (train_loader, val_loader)
    """
    train_df = patch_df.iloc[list(fold_train_indices)].reset_index(drop=True)
    val_df = patch_df.iloc[list(fold_val_indices)].reset_index(drop=True)

    train_dataset = PatchDataset(train_df, mode="train", train_transform=train_transform, val_transform=val_transform)
    val_dataset = PatchDataset(val_df, mode="val", train_transform=train_transform, val_transform=val_transform)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
