"""Dataset utilities for Thesis_Results.csv parsing and multi-view training.

Supports:
  - Thesis_Results.csv (native format from the dataset)
  - labels.csv (standard format with image_id, view, mgi, ohi, gei columns)

The primary dataset class is MultiViewPatientDataset, which groups all three
dental views (frontal, left, right) per patient and returns them as a triplet.
This prevents data leakage and mirrors the multi-view inference pipeline.
"""

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Image extension search
# ---------------------------------------------------------------------------

def _find_with_prefix(directory: Path, prefix: str) -> Optional[Path]:
    """Find an image file in directory by trying common extensions.

    Args:
        directory: Directory to search in.
        prefix: Filename prefix (without extension).

    Returns:
        Path to found image, or None if not found.
    """
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = directory / f"{prefix}{ext}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def _parse_scores_from_text(score_text: str) -> Tuple[int, int, int]:
    """Parse score text containing MGI/OHI/GEI numeric values.

    Args:
        score_text: Composite score string such as
            "MGI-0,OHI-1,GEI-0" or " MGI- 4, OHI-0,GEI-1".

    Returns:
        Tuple (mgi, ohi, gei).

    Raises:
        ValueError: If one of the score tokens cannot be extracted or is out of range.
    """
    clean = str(score_text).strip()
    mgi_match = re.search(r"MGI\s*-\s*(\d+)", clean, flags=re.IGNORECASE)
    ohi_match = re.search(r"OHI\s*-\s*(\d+)", clean, flags=re.IGNORECASE)
    gei_match = re.search(r"GEI\s*-\s*(\d+)", clean, flags=re.IGNORECASE)

    if not (mgi_match and ohi_match and gei_match):
        raise ValueError(f"Could not parse score text: {score_text!r}")

    mgi = int(mgi_match.group(1))
    ohi = int(ohi_match.group(1))
    gei = int(gei_match.group(1))

    if not (0 <= mgi <= 4 and 0 <= ohi <= 3 and 0 <= gei <= 3):
        raise ValueError(f"Out-of-range scores (mgi={mgi}, ohi={ohi}, gei={gei}) from: {score_text!r}")

    return mgi, ohi, gei


# ---------------------------------------------------------------------------
# Record loading from CSV files
# ---------------------------------------------------------------------------

def _load_from_thesis_results(data_dir: Path, labels_path: Path) -> List[Dict[str, Any]]:
    """Load image-level records using Thesis_Results.csv and F/L/R naming convention.

    Args:
        data_dir: Root directory of the Thesis_Data folder.
        labels_path: Path to Thesis_Results.csv.

    Returns:
        List of record dictionaries.
    """
    df = pd.read_csv(labels_path, usecols=[0, 1, 2], dtype=str)
    photo_root = data_dir / "Thesis_Photographs"

    view_map = {
        "frontal": (photo_root / "Frontal", "F"),
        "left":    (photo_root / "Left_Lateral", "L"),
        "right":   (photo_root / "Right_Lateral", "R"),
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
            logger.warning("Skipping malformed label row sl_no=%s name=%s: %s", sl_no_raw, name, exc)
            continue

        for view_name, (folder, prefix) in view_map.items():
            image_path = _find_with_prefix(folder, f"{prefix}{patient_id}")
            if image_path is None:
                logger.debug("Missing image for patient_id=%s view=%s", patient_id, view_name)
                continue

            records.append({
                "patient_id": str(patient_id),
                "patient_name": name,
                "view": view_name,
                "image_path": str(image_path),
                "mgi": mgi,
                "ohi": ohi,
                "gei": gei,
                "label_source": str(labels_path),
            })

    logger.info("Loaded %d image records from Thesis_Results.csv (%s)", len(records), labels_path)
    return records


def _load_from_labels_csv(data_dir: Path, labels_path: Path) -> List[Dict[str, Any]]:
    """Load image-level records from a generic labels.csv.

    Required columns: image_id, view, mgi, ohi, gei
    Optional columns: patient_id, image_path, name

    Args:
        data_dir: Root directory for resolving relative image paths.
        labels_path: Path to labels.csv.

    Returns:
        List of record dictionaries.
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
            image_path: Optional[Path] = data_dir / str(row["image_path"]).strip()
        else:
            candidate_dir = data_dir / view
            image_path = _find_with_prefix(candidate_dir, image_id)

        if image_path is None or not Path(image_path).exists():
            logger.warning("Skipping labels.csv row with unresolved image: image_id=%s", image_id)
            continue

        records.append({
            "patient_id": patient_id,
            "patient_name": str(row.get("name", "")).strip(),
            "view": view,
            "image_path": str(image_path),
            "mgi": int(row["mgi"]),
            "ohi": int(row["ohi"]),
            "gei": int(row["gei"]),
            "label_source": str(labels_path),
        })

    logger.info("Loaded %d image records from labels.csv (%s)", len(records), labels_path)
    return records


# ---------------------------------------------------------------------------
# Public dataset loader
# ---------------------------------------------------------------------------

def load_dataset(data_dir: PathLike) -> List[Dict[str, Any]]:
    """Load image-level records from either labels.csv or Thesis_Results.csv.

    Searches for labels files in data_dir and its parent directory.
    labels.csv takes priority over Thesis_Results.csv.

    Args:
        data_dir: Dataset directory path.

    Returns:
        List of dictionaries, one per image, containing:
            patient_id, view, image_path, mgi, ohi, gei, label_source.

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
            f"No labels.csv or Thesis_Results.csv found under {root} or its parent.\n"
            f"Tried: {candidates}"
        )

    if labels_path.name.lower() == "labels.csv":
        return _load_from_labels_csv(root, labels_path)

    # Thesis_Results format — use labels_path's parent as data root
    return _load_from_thesis_results(labels_path.parent, labels_path)


# ---------------------------------------------------------------------------
# Multi-view patient dataset (primary training dataset)
# ---------------------------------------------------------------------------

class MultiViewPatientDataset(Dataset):
    """Patient-level dataset returning frontal, left, right view triplets.

    Each sample contains all three views for one patient plus their labels.
    Patients missing any view are excluded to ensure consistent batch shapes.

    Args:
        records: Output of load_dataset().
        transform: Albumentations transform applied to ALL three views.
        image_size: Target image size (square). Images are resized before transform.
        require_all_views: If True, skip patients without all 3 views.
    """

    REQUIRED_VIEWS = ("frontal", "left", "right")

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        transform=None,
        image_size: int = 336,
        require_all_views: bool = True,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.image_size = image_size

        # Group by patient
        grouped: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"views": {}})
        for row in records:
            pid = str(row["patient_id"])
            grouped[pid].setdefault("patient_id", pid)
            grouped[pid].setdefault("patient_name", row.get("patient_name", ""))
            grouped[pid]["labels"] = {
                "mgi": int(row["mgi"]),
                "ohi": int(row["ohi"]),
                "gei": int(row["gei"]),
            }
            grouped[pid]["views"][str(row["view"]).lower()] = row["image_path"]

        samples: List[Dict[str, Any]] = []
        for sample in grouped.values():
            has_all = all(v in sample["views"] for v in self.REQUIRED_VIEWS)
            if require_all_views and not has_all:
                missing_views = [v for v in self.REQUIRED_VIEWS if v not in sample["views"]]
                logger.debug(
                    "Skipping patient %s — missing views: %s",
                    sample.get("patient_id"), missing_views,
                )
                continue
            samples.append(sample)

        self.samples = sorted(
            samples,
            key=lambda x: int(x["patient_id"]) if str(x["patient_id"]).isdigit() else 0,
        )

        # Build flat label arrays for WeightedRandomSampler
        self.mgi_labels = [s["labels"]["mgi"] for s in self.samples]
        self.ohi_labels = [s["labels"]["ohi"] for s in self.samples]
        self.gei_labels = [s["labels"]["gei"] for s in self.samples]

        logger.info(
            "MultiViewPatientDataset: %d patients (all-view requirement=%s)",
            len(self.samples), require_all_views,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_and_preprocess(self, image_path: str) -> np.ndarray:
        """Load image, resize, apply CLAHE + white balance.

        Args:
            image_path: Path to source image file.

        Returns:
            RGB uint8 array of shape (H, W, 3) ready for transform.
        """
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            logger.warning("Could not load image %s — returning black frame", image_path)
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Resize
        img_bgr = cv2.resize(
            img_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4
        )
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Gray-world white balance
        avg_r = img_rgb[:, :, 0].mean()
        avg_g = img_rgb[:, :, 1].mean()
        avg_b = img_rgb[:, :, 2].mean()
        gray_avg = (avg_r + avg_g + avg_b) / 3.0

        wb = img_rgb.astype(np.float32)
        if avg_r > 1e-6:
            wb[:, :, 0] = np.clip(wb[:, :, 0] * (gray_avg / avg_r), 0, 255)
        if avg_g > 1e-6:
            wb[:, :, 1] = np.clip(wb[:, :, 1] * (gray_avg / avg_g), 0, 255)
        if avg_b > 1e-6:
            wb[:, :, 2] = np.clip(wb[:, :, 2] * (gray_avg / avg_b), 0, 255)
        img_wb = wb.astype(np.uint8)

        # CLAHE on L channel
        lab = cv2.cvtColor(img_wb, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        img_clahe = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2RGB)

        return img_clahe

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
        """Return a patient triplet.

        Returns:
            Tuple of (frontal_tensor, left_tensor, right_tensor, labels_dict).
        """
        sample = self.samples[index]
        views = sample["views"]
        labels = sample["labels"]

        tensors = []
        for view_name in self.REQUIRED_VIEWS:
            path = views.get(view_name, views.get(list(views.keys())[0]))  # fallback to any view
            img = self._load_and_preprocess(str(path))

            if self.transform is not None:
                img = self.transform(image=img)["image"]
            else:
                # Manual conversion to tensor
                img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

            tensors.append(img)

        return tensors[0], tensors[1], tensors[2], labels


# ---------------------------------------------------------------------------
# WeightedRandomSampler builder
# ---------------------------------------------------------------------------

def build_weighted_sampler(
    dataset: MultiViewPatientDataset,
    task: str = "mgi",
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler to equalise class frequency per epoch.

    Args:
        dataset: MultiViewPatientDataset instance.
        task: Task to stratify on ('mgi', 'ohi', or 'gei').

    Returns:
        WeightedRandomSampler that over-samples minority classes.
    """
    labels = getattr(dataset, f"{task}_labels")
    class_counts: Dict[int, int] = {}
    for lbl in labels:
        class_counts[lbl] = class_counts.get(lbl, 0) + 1

    sample_weights = [1.0 / class_counts[lbl] for lbl in labels]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )
    logger.info(
        "WeightedRandomSampler built for task=%s. Class counts: %s",
        task, dict(sorted(class_counts.items())),
    )
    return sampler


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def get_dataloaders(
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    dataset: MultiViewPatientDataset,
    val_dataset: MultiViewPatientDataset,
    batch_size: int,
    num_workers: int = 0,
    use_weighted_sampler: bool = True,
    sampler_task: str = "mgi",
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders for one fold.

    Args:
        train_indices: Patient indices for the train split.
        val_indices: Patient indices for the val split (unused — val_dataset used directly).
        dataset: Full training MultiViewPatientDataset (with training transforms).
        val_dataset: Validation MultiViewPatientDataset (with val transforms).
        batch_size: Batch size.
        num_workers: DataLoader workers.
        use_weighted_sampler: If True, use WeightedRandomSampler in train loader.
        sampler_task: Task used to compute sample weights.

    Returns:
        (train_dataloader, val_dataloader)
    """
    from torch.utils.data import Subset

    train_subset = Subset(dataset, list(train_indices))
    val_subset = Subset(val_dataset, list(val_indices))

    pin_memory = torch.cuda.is_available()

    if use_weighted_sampler:
        # Build per-subset sampler
        subset_labels = [dataset.samples[i]["labels"][sampler_task] for i in train_indices]
        class_counts: Dict[int, int] = {}
        for lbl in subset_labels:
            class_counts[lbl] = class_counts.get(lbl, 0) + 1
        sample_weights = [1.0 / class_counts[lbl] for lbl in subset_labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Legacy PatchDataset (kept for backwards compatibility)
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    """Patch-level dataset reading from patches_labels.csv.

    Kept for backwards compatibility with patch-based training approaches.
    Prefer MultiViewPatientDataset for new training runs.
    """

    def __init__(
        self,
        patch_df: Union[pd.DataFrame, PathLike],
        mode: str,
        train_transform=None,
        val_transform=None,
    ) -> None:
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
        image_bgr = cv2.imread(str(row["patch_path"]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            image_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image_rgb)
            patch_tensor = transformed["image"]
        else:
            patch_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0

        return patch_tensor, {
            "mgi": int(row["mgi"]),
            "ohi": int(row["ohi"]),
            "gei": int(row["gei"]),
        }
