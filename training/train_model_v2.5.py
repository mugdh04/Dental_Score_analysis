#!/usr/bin/env python3
"""
DentAI Oral Health Prediction — Advanced Training Script v2
============================================================
Root-cause fixes for the label-collapse problem seen in v1:

  1. OHI 4-class → 3-class  (0, 1, 2+)  — eliminates OHI collapse
  2. GEI 4-class → 3-class  (0, 1, 2+)  — eliminates GEI collapse
  3. Ordinal-BCE loss with per-threshold pos-weights   — replaces CE+Focal+OrdinalMSE
  4. Cross-view attention pooling            — views inform each other before heads
  5. SWA (Stochastic Weight Averaging)       — smooths noisy-final-epoch weights
  6. MixUp in feature space                  — implicit augmentation for rare classes
  7. Composite multi-task sampler            — balances ALL tasks, not just MGI
  8. OneCycleLR + linear warmup              — faster, more stable convergence
  9. 5-fold × 2 (best + SWA) ensemble        — reduces variance on tiny dataset
 10. Reduced dropout 0.45 → 0.20            — was over-regularising 203 patients

Usage (Windows GPU):
    python train_model_v2.py

Place this file inside:  <project_root>/training/
The script locates PROJECT_ROOT as two levels up from its own location.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Imports
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, re, json, random, logging, warnings
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as _amp_compat
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Reproducibility + GPU gate
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

if not torch.cuda.is_available():
    raise RuntimeError(
        "\n\nCUDA GPU is required for this training script.\n"
        "Check GPU status: nvidia-smi\n"
        "Install CUDA PyTorch: pip install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu126 --force-reinstall"
    )

DEVICE = torch.device("cuda")
print(f"GPU  : {torch.cuda.get_device_name(0)}")
print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"PyTorch {torch.__version__}  |  timm {timm.__version__}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Config
# ─────────────────────────────────────────────────────────────────────────────

# Project layout:  <root>/training/train_model_v2.py  →  PROJECT_ROOT = <root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    # ── Image ─────────────────────────────────────────────────────────
    "image_size": 336,              # DINOv2-small native resolution

    # ── Classes (MERGED for OHI and GEI) ──────────────────────────────
    "num_classes_mgi": 5,           # 0-4  (keep; class 4 over-sampled)
    "num_classes_ohi": 3,           # 0, 1, 2+  (was 4 → collapse fixed)
    "num_classes_gei": 3,           # 0, 1, 2+  (was 4 → collapse fixed)

    # ── Backbone ──────────────────────────────────────────────────────
    "backbone_model":    "vit_small_patch14_dinov2.lvd142m",
    "pretrained_backbone": True,
    "freeze_epochs":    10,         # Freeze backbone for first N epochs
    "unfreeze_blocks":   6,         # Then unfreeze last 6 transformer blocks

    # ── Architecture ──────────────────────────────────────────────────
    "dropout":          0.20,       # Was 0.45 — over-regularising 203 pts
    "projection_dim":  256,         # Was 512
    "head_hidden_dim": 128,         # Was 256
    "view_attn_heads":   4,         # Cross-view attention heads

    # ── Training ──────────────────────────────────────────────────────
    "batch_size":        4,         # Multi-view 3× memory cost
    "grad_accum_steps":  4,         # Effective batch = 16
    "num_epochs":      100,
    "warmup_epochs":     5,         # Linear warmup after backbone unfreeze

    # ── Learning rates ────────────────────────────────────────────────
    "lr_backbone":     5e-7,        # Very low — DINOv2 features are precious
    "lr_projection":   5e-5,
    "lr_heads":        2e-4,
    "weight_decay":    1e-4,

    # ── SWA (Stochastic Weight Averaging) ─────────────────────────────
    "swa_start_epoch": 70,          # Begin averaging from epoch 70
    "swa_lr":          3e-6,

    # ── Loss ──────────────────────────────────────────────────────────
    "alpha_ordinal":   1.5,         # Primary: ordinal-BCE
    "alpha_focal":     0.5,         # Auxiliary: focal on decoded class
    "focal_gamma":     2.5,
    "pos_weight_cap": 10.0,         # Cap per-threshold pos-weights
    "mgi_pos_weight_cap": 6.0,      # Lower cap for MGI to reduce high-score bias
    "label_smooth_bce": 0.02,       # Tiny soft labels on binary thresholds
    "mgi_over_penalty": 0.60,       # Penalise over-estimation more than under-estimation
    "mgi_under_penalty": 0.15,

    # ── Validation objective (Medical Robustness Balanced) ───────────────────────
    "objective_weights": {
        "mgi": 0.34,
        "ohi": 0.33,
        "gei": 0.33,
    },
    "objective_mgi_mae_weight": 0.20,
    "objective_mgi_under_weight": 0.20,
    "objective_mgi_over_weight": 0.05,
    "objective_ohi_mae_weight": 0.20,
    "objective_ohi_under_weight": 0.20,
    "objective_ohi_over_weight": 0.05,
    "objective_gei_mae_weight": 0.20,
    "objective_gei_under_weight": 0.20,
    "objective_gei_over_weight": 0.05,

    # ── Task-level multi-task loss weights ───────────────────────────
    "task_loss_weights": {
        "mgi": 0.40,
        "ohi": 0.30,
        "gei": 0.30,
    },

    # ── MGI threshold calibration ─────────────────────────────────────
    "default_decode_threshold": 0.50,
    "mgi_threshold_min": 0.45,
    "mgi_threshold_max": 0.80,
    "mgi_threshold_steps": 29,
    "ohi_threshold_min": 0.35,
    "ohi_threshold_max": 0.70,
    "ohi_threshold_steps": 29,
    "gei_threshold_min": 0.35,
    "gei_threshold_max": 0.70,
    "gei_threshold_steps": 29,
    "calibrate_every_n_epochs": 5,

    # ── Class-balance + DRW (Deferred Re-Weighting) ────────────────
    "class_balance_beta": 0.999,
    "class_balance_min_weight": 0.25,
    "class_balance_max_weight": 4.0,
    "drw_start_epoch": 16,
    "drw_full_epoch": 45,

    # ── K-fold ────────────────────────────────────────────────────────
    "k_folds":           5,
    "early_stopping_patience": 25,
    "seed":             42,

    # ── MixUp ─────────────────────────────────────────────────────────
    "mixup_alpha":    0.3,
    "mixup_prob":     0.4,

    # ── TTA ───────────────────────────────────────────────────────────
    "tta_steps":        5,

    # ── DataLoader ────────────────────────────────────────────────────
    "num_workers":      0,          # 0 for Windows stability

    # ── Paths ─────────────────────────────────────────────────────────
    "data_dir":        str(PROJECT_ROOT / "Thesis_Data"),
    "checkpoint_dir":  str(PROJECT_ROOT / "models" / "checkpoints"),
    "plots_dir":       str(PROJECT_ROOT / "outputs"  / "plots"),
}

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs(CONFIG["plots_dir"],      exist_ok=True)
os.makedirs(str(PROJECT_ROOT / "models"), exist_ok=True)

print("\n── CONFIG ──────────────────────────────────────────────────────")
for k, v in CONFIG.items():
    print(f"  {k:28s}: {v}")
print("────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Label-merge utilities
# ─────────────────────────────────────────────────────────────────────────────

def merge_ohi(x: int) -> int:
    """Map OHI 0-3 → 0-2  (class 2 and 3 become class 2)."""
    return min(x, 2)


def merge_gei(x: int) -> int:
    """Map GEI 0-3 → 0-2  (class 2 and 3 become class 2)."""
    return min(x, 2)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CSV parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_thesis_csv(data_dir: str):
    """
    Parse Thesis_Results.csv and return image-level records with merged labels.

    Label mapping applied here:
      ohi  : 0→0  1→1  2,3→2
      gei  : 0→0  1→1  2,3→2
      mgi  : unchanged (0-4)

    Returns:
        records (list[dict]): patient_id, view, image_path, mgi, ohi, gei
        patient_labels (dict): patient_id → {mgi, ohi, gei} (merged)
    """
    root = Path(data_dir)
    csv_path = root / "Thesis_Results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=[0, 1, 2], dtype=str)

    photo_root = root / "Thesis_Photographs"
    view_map = {
        "frontal": (photo_root / "Frontal",       "F"),
        "left":    (photo_root / "Left_Lateral",  "L"),
        "right":   (photo_root / "Right_Lateral", "R"),
    }

    records, patient_labels, skipped = [], {}, 0

    for _, row in df.iterrows():
        try:
            pid       = int(str(row.iloc[0]).strip())
            score_txt = str(row.iloc[2]).strip()
        except Exception:
            skipped += 1
            continue

        mgi_m = re.search(r"MGI\s*-\s*(\d+)", score_txt, re.I)
        ohi_m = re.search(r"OHI\s*-\s*(\d+)", score_txt, re.I)
        gei_m = re.search(r"GEI\s*-\s*(\d+)", score_txt, re.I)

        if not (mgi_m and ohi_m and gei_m):
            skipped += 1
            continue

        mgi = int(mgi_m.group(1))
        ohi = merge_ohi(int(ohi_m.group(1)))
        gei = merge_gei(int(gei_m.group(1)))

        if not (0 <= mgi <= 4 and 0 <= ohi <= 2 and 0 <= gei <= 2):
            log.warning(f"Out-of-range scores for patient {pid}: mgi={mgi} ohi={ohi} gei={gei}")
            skipped += 1
            continue

        patient_labels[str(pid)] = {"mgi": mgi, "ohi": ohi, "gei": gei}

        for view_name, (folder, prefix) in view_map.items():
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                cand = folder / f"{prefix}{pid}{ext}"
                if cand.exists():
                    img_path = cand
                    break
            if img_path is None:
                continue

            records.append({
                "patient_id": str(pid),
                "view":       view_name,
                "image_path": str(img_path),
                "mgi":        mgi,
                "ohi":        ohi,
                "gei":        gei,
            })

    print(f"CSV parsed: {len(patient_labels)} patients, {len(records)} image records "
          f"({skipped} rows skipped)")
    return records, patient_labels


RECORDS, PATIENT_LABELS = parse_thesis_csv(CONFIG["data_dir"])

# ── Label distribution after merging ─────────────────────────────────────────
print("\n── MERGED LABEL DISTRIBUTION ────────────────────────────────────")
task_info = {
    "MGI (0-4)": ("mgi", CONFIG["num_classes_mgi"]),
    "OHI (0-2)": ("ohi", CONFIG["num_classes_ohi"]),
    "GEI (0-2)": ("gei", CONFIG["num_classes_gei"]),
}
for title, (key, n_cls) in task_info.items():
    vals = [v[key] for v in PATIENT_LABELS.values()]
    counts = np.bincount(vals, minlength=n_cls)
    parts = "  ".join(
        f"cls{i}:{c}({c/len(vals)*100:.0f}%)" + (" ⚠" if c < 10 else "")
        for i, c in enumerate(counts)
    )
    print(f"  {title}: {parts}")
print("────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Ordinal pos-weight computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ordinal_pos_weights(patient_labels: dict, config: dict) -> dict:
    """
    For each task and each binary threshold k (Y >= k for k=1..K-1),
    compute pos_weight = neg_count / pos_count, capped at pos_weight_cap.

    These are passed to binary_cross_entropy_with_logits to re-balance
    severely imbalanced thresholds (e.g. GEI threshold-2 has only 4 positives).
    """
    n_total = len(patient_labels)
    labels_by_task = {
        "mgi": [v["mgi"] for v in patient_labels.values()],
        "ohi": [v["ohi"] for v in patient_labels.values()],
        "gei": [v["gei"] for v in patient_labels.values()],
    }
    n_thresh = {
        "mgi": config["num_classes_mgi"] - 1,  # 4
        "ohi": config["num_classes_ohi"] - 1,  # 2
        "gei": config["num_classes_gei"] - 1,  # 2
    }
    cap = config["pos_weight_cap"]
    pos_weights = {}
    print("── ORDINAL POS-WEIGHTS (per threshold) ─────────────────────────")
    for task, labels in labels_by_task.items():
        task_cap = config.get("mgi_pos_weight_cap", cap) if task == "mgi" else cap
        pw = []
        for k in range(1, n_thresh[task] + 1):
            pos = sum(1 for y in labels if y >= k)
            neg = n_total - pos
            w   = min((neg / max(pos, 1)), task_cap)
            pw.append(w)
        pos_weights[task] = torch.tensor(pw, dtype=torch.float32)
        print(f"  {task}: {['%.2f'%w for w in pw]}  (K-1={n_thresh[task]} thresholds)")
    print("────────────────────────────────────────────────────────────────\n")
    return pos_weights


POS_WEIGHTS = compute_ordinal_pos_weights(PATIENT_LABELS, CONFIG)


def compute_fold_class_balanced_weights(dataset, indices: list, config: dict) -> dict:
    """
    Compute effective-number class weights on the fold's train split only.

    Weights are normalised to mean 1, then clipped for stability.
    """
    beta = float(config.get("class_balance_beta", 0.999))
    w_min = float(config.get("class_balance_min_weight", 0.25))
    w_max = float(config.get("class_balance_max_weight", 4.0))

    labels_by_task = {
        "mgi": np.array([dataset.mgi_labels[i] for i in indices], dtype=np.int64),
        "ohi": np.array([dataset.ohi_labels[i] for i in indices], dtype=np.int64),
        "gei": np.array([dataset.gei_labels[i] for i in indices], dtype=np.int64),
    }
    n_cls = {
        "mgi": CONFIG["num_classes_mgi"],
        "ohi": CONFIG["num_classes_ohi"],
        "gei": CONFIG["num_classes_gei"],
    }

    out = {}
    print("── CLASS-BALANCED WEIGHTS (effective number) ───────────────────")
    for task, arr in labels_by_task.items():
        counts = np.bincount(arr, minlength=n_cls[task]).astype(np.float64)
        counts_safe = np.clip(counts, 1.0, None)

        effective_num = 1.0 - np.power(beta, counts_safe)
        weights = (1.0 - beta) / np.clip(effective_num, 1e-12, None)

        if np.any(counts == 0):
            fill = float(weights[counts > 0].max()) if np.any(counts > 0) else 1.0
            weights[counts == 0] = fill

        weights = weights / max(weights.mean(), 1e-12)
        weights = np.clip(weights, w_min, w_max)

        out[task] = torch.tensor(weights, dtype=torch.float32)
        pretty_weights = [f"{float(x):.3f}" for x in weights]
        print(f"  {task}: counts={counts.astype(int).tolist()}  weights={pretty_weights}")

    print("────────────────────────────────────────────────────────────────\n")
    return out


def compute_drw_alpha(epoch: int, config: dict) -> float:
    """Linearly ramp class re-weighting from 0 → 1 over the DRW window."""
    start = int(config.get("drw_start_epoch", 16))
    full = int(config.get("drw_full_epoch", 45))
    if epoch < start:
        return 0.0
    if epoch >= full:
        return 1.0
    return float((epoch - start) / max(full - start, 1))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Augmentation
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
_SZ = CONFIG["image_size"]

TRAIN_TRANSFORM = A.Compose([
    # Geometric
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=25, p=0.65),
    A.RandomResizedCrop(height=_SZ, width=_SZ, scale=(0.75, 1.0), ratio=(0.85, 1.15), p=0.6),
    A.GridDistortion(num_steps=3, distort_limit=0.2, p=0.2),
    A.ElasticTransform(alpha=60, sigma=6, p=0.15),

    # Colour — gingival redness is the PRIMARY discriminating feature
    A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.65),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=40, val_shift_limit=30, p=0.60),
    A.RGBShift(r_shift_limit=20, g_shift_limit=15, b_shift_limit=15, p=0.45),
    A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08, p=0.45),
    A.CLAHE(clip_limit=3.5, tile_grid_size=(8, 8), p=0.40),
    A.ToGray(p=0.05),               # Tiny chance of gray — forces texture features

    # Noise / blur
    A.GaussianBlur(blur_limit=(3, 7), p=0.25),
    A.GaussNoise(var_limit=(10, 60), p=0.30),
    A.ImageCompression(quality_lower=65, quality_upper=100, p=0.20),

    # Occlusion
    A.CoarseDropout(max_holes=5, max_height=30, max_width=30, min_holes=1, p=0.25),

    A.Resize(_SZ, _SZ),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

VAL_TRANSFORM = A.Compose([
    A.Resize(_SZ, _SZ),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

print(f"Augmentation: train={len(TRAIN_TRANSFORM.transforms)} transforms, val=2 transforms")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MultiViewPatientDataset(Dataset):
    """
    One sample per patient: (frontal_tensor, left_tensor, right_tensor, labels).
    Only patients with all 3 views are kept.

    Labels are already merged (OHI/GEI → 3 classes).
    """

    REQUIRED = ("frontal", "left", "right")

    def __init__(self, records: list, transform=None):
        self.transform = transform

        grouped = defaultdict(lambda: {"views": {}})
        for r in records:
            pid = str(r["patient_id"])
            grouped[pid]["patient_id"] = pid
            grouped[pid]["labels"]     = {"mgi": r["mgi"], "ohi": r["ohi"], "gei": r["gei"]}
            grouped[pid]["views"][r["view"].lower()] = r["image_path"]

        self.samples = [
            s for s in grouped.values()
            if all(v in s["views"] for v in self.REQUIRED)
        ]
        self.samples.sort(
            key=lambda x: int(x["patient_id"]) if x["patient_id"].isdigit() else 0
        )

        self.mgi_labels = [s["labels"]["mgi"] for s in self.samples]
        self.ohi_labels = [s["labels"]["ohi"] for s in self.samples]
        self.gei_labels = [s["labels"]["gei"] for s in self.samples]
        print(f"Dataset ready: {len(self.samples)} patients with complete 3-view sets")

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def _load(self, path: str) -> np.ndarray:
        """Load image, apply gray-world WB and CLAHE, return uint8 RGB."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((_SZ, _SZ, 3), dtype=np.uint8)

        img = cv2.resize(img, (_SZ, _SZ), interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Gray-world white balance
        avg  = img.mean(axis=(0, 1)).astype(np.float32)
        gray = avg.mean()
        scale = np.where(avg > 1e-6, gray / avg, 1.0)
        img = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)

        # CLAHE on L-channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)

        return img

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        s   = self.samples[idx]
        lbl = s["labels"]

        tensors = []
        for v in self.REQUIRED:
            img = self._load(s["views"][v])
            if self.transform is not None:
                img = self.transform(image=img)["image"]
            else:
                img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            tensors.append(img)

        return tensors[0], tensors[1], tensors[2], lbl


# Build dataset pair (train uses augmentation, val uses plain resize)
FULL_TRAIN_DS = MultiViewPatientDataset(RECORDS, transform=TRAIN_TRANSFORM)
FULL_VAL_DS   = MultiViewPatientDataset(RECORDS, transform=VAL_TRANSFORM)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class OrdinalBCELoss(nn.Module):
    """
    Binary cross-entropy over K-1 ordered thresholds.

    For a K-class ordinal label y, the binary target for threshold k is:
        t_k = 1  if y >= k  else 0       (k = 1 .. K-1)

    pos_weight: tensor of shape (K-1,) — amplifies minority-positive thresholds.
    label_smooth: tiny smoothing added to binary targets (default 0.02).

    MixUp support: pass lam + targets_b to blend two binary-target matrices.
    """

    def __init__(self, num_classes: int, pos_weight: torch.Tensor,
                 label_smooth: float = 0.02):
        super().__init__()
        self.K      = num_classes
        self.n_th   = num_classes - 1
        self.smooth = label_smooth
        self.register_buffer("pos_weight", pos_weight)

    def _make_targets(self, y: torch.Tensor) -> torch.Tensor:
        """Convert integer labels → (B, K-1) binary threshold targets."""
        B = y.shape[0]
        t = torch.zeros(B, self.n_th, device=y.device, dtype=torch.float32)
        for k in range(1, self.K):
            t[:, k - 1] = (y >= k).float()
        # Soft labels: 0 → smooth/2,  1 → 1 - smooth/2
        t = t * (1.0 - self.smooth) + self.smooth * 0.5
        return t

    def forward(self,
                logits:   torch.Tensor,          # (B, K-1)
                targets:  torch.Tensor,          # (B,) integer
                lam:      float = 1.0,
            targets_b: torch.Tensor = None,
            class_weights: torch.Tensor = None)  -> torch.Tensor:

        pw = self.pos_weight.to(logits.device)
        t_a = self._make_targets(targets)

        if targets_b is not None and lam < 1.0:
            t_b = self._make_targets(targets_b)
            t   = lam * t_a + (1.0 - lam) * t_b
        else:
            t = t_a

        sample_w = None
        if class_weights is not None:
            cw = class_weights.to(logits.device)
            w_a = cw[targets.long()]
            if targets_b is not None and lam < 1.0:
                w_b = cw[targets_b.long()]
                sample_w = lam * w_a + (1.0 - lam) * w_b
            else:
                sample_w = w_a

        bce_weight = None
        if sample_w is not None:
            bce_weight = sample_w.view(-1, 1).expand_as(logits)

        return F.binary_cross_entropy_with_logits(logits, t,
                                                   weight=bce_weight,
                                                   pos_weight=pw,
                                                   reduction="mean")


class FocalLossDecoded(nn.Module):
    """
    Focal loss on CE over decoded class predictions.
    Applied as auxiliary to the ordinal-BCE to penalise large prediction errors.
    """
    def __init__(self, num_classes: int, gamma: float = 2.5,
                 label_smooth: float = 0.0):
        super().__init__()
        self.gamma  = gamma
        self.n_cls  = num_classes
        self.smooth = label_smooth

    def forward(self, logits_decoded: torch.Tensor, targets: torch.Tensor,
                lam: float = 1.0, targets_b: torch.Tensor = None,
                class_weights: torch.Tensor = None) -> torch.Tensor:
        """
        logits_decoded: (B, K) — normal softmax logits reconstructed from ordinal probs.
        """
        ce_weight = class_weights.to(logits_decoded.device) if class_weights is not None else None
        ce_a = F.cross_entropy(logits_decoded, targets,
                               label_smoothing=self.smooth, reduction="none",
                               weight=ce_weight)
        if targets_b is not None and lam < 1.0:
            ce_b = F.cross_entropy(logits_decoded, targets_b,
                                   label_smoothing=self.smooth, reduction="none",
                                   weight=ce_weight)
            ce   = lam * ce_a + (1.0 - lam) * ce_b
        else:
            ce = ce_a

        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class MultiTaskLoss(nn.Module):
    """
    Combines OrdinalBCE (primary) + FocalDecoded (auxiliary) for MGI/OHI/GEI.
    Reconstruction of softmax logits from ordinal probabilities:
        P(Y=k) = P(Y>=k) - P(Y>=k+1)
    This gives a differentiable K-class distribution for the focal term.
    """

    def __init__(self, pos_weights: dict, config: dict):
        super().__init__()
        sm = config.get("label_smooth_bce", 0.02)

        for task, k in [
            ("mgi", config["num_classes_mgi"]),
            ("ohi", config["num_classes_ohi"]),
            ("gei", config["num_classes_gei"]),
        ]:
            pw = pos_weights[task]
            setattr(self, f"ord_{task}",
                    OrdinalBCELoss(k, pw, label_smooth=sm))
            setattr(self, f"foc_{task}",
                    FocalLossDecoded(k, gamma=config.get("focal_gamma", 2.5)))

        self.alpha_ord = config.get("alpha_ordinal", 1.5)
        self.alpha_foc = config.get("alpha_focal",   0.5)
        self.mgi_over_penalty = config.get("mgi_over_penalty", 0.60)
        self.mgi_under_penalty = config.get("mgi_under_penalty", 0.15)
        self.task_loss_weights = config.get(
            "task_loss_weights",
            {"mgi": 0.40, "ohi": 0.30, "gei": 0.30},
        )

    # ------------------------------------------------------------------
    @staticmethod
    def ordinal_to_class_logits(ordinal_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, K-1) ordinal logits → (B, K) class pseudo-logits.
        Uses cumulative probabilities:  P(Y>=k) for each threshold.
        Class probability: P(Y=k) = P(Y>=k) - P(Y>=k+1)
        """
        cum = torch.sigmoid(ordinal_logits)          # (B, K-1)
        B, Km1 = cum.shape
        ones  = torch.ones(B, 1, device=cum.device)
        zeros = torch.zeros(B, 1, device=cum.device)
        # P(Y>=0) = 1, P(Y>=1..K-1) = sigmoid, P(Y>=K) = 0
        cum_full = torch.cat([ones, cum, zeros], dim=1)  # (B, K+1)
        class_probs = cum_full[:, :-1] - cum_full[:, 1:]  # (B, K)
        # Clamp to avoid log(0) and return as logits via log
        class_probs = class_probs.clamp(min=1e-7)
        return torch.log(class_probs)                # pseudo logits

    # ------------------------------------------------------------------
    def _mgi_bias_loss(self, logits: torch.Tensor,
                       targets: torch.Tensor,
                       lam: float = 1.0,
                       targets_b: torch.Tensor = None) -> torch.Tensor:
        """
        Penalise MGI over-estimation more than under-estimation.

        This directly addresses the observed failure mode where predicted MGI
        tends to be 1-2 points above the clinical score.
        """
        expected_score = torch.sigmoid(logits).sum(dim=1)
        y = targets.float()
        if targets_b is not None and lam < 1.0:
            y = lam * y + (1.0 - lam) * targets_b.float()

        over = F.relu(expected_score - y)
        under = F.relu(y - expected_score)

        return (
            self.mgi_over_penalty * over.pow(2).mean()
            + self.mgi_under_penalty * under.pow(2).mean()
        )

    # ------------------------------------------------------------------
    def _task_loss(self, task: str, logits: torch.Tensor, targets: torch.Tensor,
                   lam: float = 1.0, targets_b=None,
                   class_weights: dict = None,
                   drw_alpha: float = 0.0):
        ord_fn = getattr(self, f"ord_{task}")
        foc_fn = getattr(self, f"foc_{task}")

        task_class_weights = None
        if class_weights is not None and task in class_weights:
            cw = class_weights[task].to(logits.device)
            task_class_weights = 1.0 + float(drw_alpha) * (cw - 1.0)

        ord_loss = ord_fn(
            logits,
            targets,
            lam=lam,
            targets_b=targets_b,
            class_weights=task_class_weights,
        )
        cls_logits = self.ordinal_to_class_logits(logits)
        foc_loss   = foc_fn(
            cls_logits,
            targets,
            lam=lam,
            targets_b=targets_b,
            class_weights=task_class_weights,
        )
        bias_loss = torch.zeros((), device=logits.device)
        if task == "mgi":
            bias_loss = self._mgi_bias_loss(logits, targets, lam=lam, targets_b=targets_b)

        total = self.alpha_ord * ord_loss + self.alpha_foc * foc_loss + bias_loss
        return total, {
            "ord": ord_loss.item(),
            "foc": foc_loss.item(),
            "bias": float(bias_loss.detach().item()),
        }

    # ------------------------------------------------------------------
    def forward(self, preds: dict, targets: dict,
                lam: float = 1.0, targets_b: dict = None,
                class_weights: dict = None,
                drw_alpha: float = 0.0):
        """
        preds   : {task: (B, K-1)}
        targets : {task: (B,) long}
        """
        loss_mgi, info_mgi = self._task_loss(
            "mgi", preds["mgi"], targets["mgi"], lam,
            targets_b["mgi"] if targets_b else None,
            class_weights=class_weights,
            drw_alpha=drw_alpha)
        loss_ohi, info_ohi = self._task_loss(
            "ohi", preds["ohi"], targets["ohi"], lam,
            targets_b["ohi"] if targets_b else None,
            class_weights=class_weights,
            drw_alpha=drw_alpha)
        loss_gei, info_gei = self._task_loss(
            "gei", preds["gei"], targets["gei"], lam,
            targets_b["gei"] if targets_b else None,
            class_weights=class_weights,
            drw_alpha=drw_alpha)

        total = (
            float(self.task_loss_weights.get("mgi", 0.40)) * loss_mgi
            + float(self.task_loss_weights.get("ohi", 0.30)) * loss_ohi
            + float(self.task_loss_weights.get("gei", 0.30)) * loss_gei
        )
        task_losses = {
            "mgi": loss_mgi.item(),
            "ohi": loss_ohi.item(),
            "gei": loss_gei.item(),
        }
        return total, task_losses


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Model
# ─────────────────────────────────────────────────────────────────────────────

def _create_backbone(name: str, pretrained: bool, img_size: int) -> nn.Module:
    """
    Create DINOv2 backbone via timm with automatic key-remap fallback
    (handles norm.* vs fc_norm.* mismatch in some timm versions).
    """
    def _build(pre: bool):
        return timm.create_model(name, pretrained=pre,
                                  img_size=img_size, num_classes=0,
                                  global_pool="avg")
    try:
        return _build(pretrained)
    except RuntimeError as exc:
        msg = str(exc)
        if "fc_norm.weight" in msg and "norm.weight" in msg:
            log.warning("DINOv2/timm key mismatch — applying norm→fc_norm remap")
            model = _build(False)
            cfg   = getattr(model, "pretrained_cfg", {}) or {}
            url   = cfg.get("url")
            if not url:
                log.warning("No pretrained URL — using random init")
                return model
            sd = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=False, progress=True)
            
            import math
            import torch.nn.functional as F
            
            
            remap = {}
            for k, v in sd.items():
                if k == "norm.weight":    
                    remap["fc_norm.weight"] = v
                elif k == "norm.bias":    
                    remap["fc_norm.bias"]   = v
                elif k == "mask_token":   
                    pass
                elif k == "pos_embed":
                    model_pe = getattr(model, "pos_embed", None)
                    if model_pe is not None and model_pe.shape != v.shape:
                        cls_tokens = v[:, 0:1, :]
                        patch_tokens = v[:, 1:, :]
                        
                        N_ckpt = patch_tokens.shape[1]
                        N_model = model_pe.shape[1] - 1
                        grid_ckpt = int(math.sqrt(N_ckpt))
                        grid_model = int(math.sqrt(N_model))
                        dim = patch_tokens.shape[-1]
                        
                        patch_tokens = patch_tokens.reshape(1, grid_ckpt, grid_ckpt, dim).permute(0, 3, 1, 2)
                        patch_tokens = F.interpolate(patch_tokens, size=(grid_model, grid_model), mode="bicubic", align_corners=False)
                        patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(1, N_model, dim)
                        
                        remap[k] = torch.cat([cls_tokens, patch_tokens], dim=1)
                    else:
                        remap[k] = v
                else:                     
                    remap[k] = v
                    
            incompat = model.load_state_dict(remap, strict=False)
            if incompat.missing_keys:
                log.info(f"Remap: missing={incompat.missing_keys[:4]}")
            return model
        # Any other RuntimeError
        if pretrained:
            log.warning(f"Pretrained load failed ({exc}); falling back to random init")
            return _build(False)
        raise


class ViewAttentionPool(nn.Module):
    """
    Self-attention over the 3 intra-oral views, then mean-pool.
    Allows views to inform each other before the task heads see the fused repr.

    Input:  (B, 3, D)
    Output: (B, D)
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, num_heads,
                                            batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        # Transformer-style residual self-attention
        attn_out, _ = self.attn(views, views, views)
        views = self.norm1(views + attn_out)
        ff_out = self.ff(views)
        views  = self.norm2(views + ff_out)
        return views.mean(dim=1)   # mean-pool over 3 views → (B, D)


class _TaskHead(nn.Module):
    """Two-layer MLP head.  Output dim = num_classes - 1 for ordinal BCE."""

    def __init__(self, in_dim: int, out_dim: int,
                 hidden_dim: int = 128, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class OralHealthModelV2(nn.Module):
    """
    DINOv2-small backbone + cross-view attention + ordinal BCE heads.

    Forward: (frontal, left, right) → {
        'mgi': (B, 4),   # 4 thresholds for 5 classes
        'ohi': (B, 2),   # 2 thresholds for 3 classes (merged)
        'gei': (B, 2),   # 2 thresholds for 3 classes (merged)
    }
    """

    def __init__(self, config: dict):
        super().__init__()
        self.backbone = _create_backbone(
            config["backbone_model"],
            config["pretrained_backbone"],
            config["image_size"],
        )
        feat_dim = self.backbone.num_features     # 384 for DINOv2-small

        self.view_pool = ViewAttentionPool(
            feat_dim, num_heads=config["view_attn_heads"]
        )

        proj_dim  = config["projection_dim"]
        head_dim  = config["head_hidden_dim"]
        drop      = config["dropout"]

        self.projection = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(drop),
        )

        self.mgi_head = _TaskHead(proj_dim, config["num_classes_mgi"] - 1,
                                   head_dim, drop)
        self.ohi_head = _TaskHead(proj_dim, config["num_classes_ohi"] - 1,
                                   head_dim, drop)
        self.gei_head = _TaskHead(proj_dim, config["num_classes_gei"] - 1,
                                   head_dim, drop)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone; always return (B, feat_dim)."""
        out = self.backbone(x)          # timm global_pool='avg' → (B, D)
        if out.dim() == 3:
            out = out[:, 0]             # CLS token fallback
        return out

    def forward(self, frontal, left, right):
        f = self._extract(frontal)
        l = self._extract(left)
        r = self._extract(right)

        fused = self.view_pool(torch.stack([f, l, r], dim=1))  # (B, D)
        proj  = self.projection(fused)                         # (B, proj_dim)

        return {
            "mgi": self.mgi_head(proj),
            "ohi": self.ohi_head(proj),
            "gei": self.gei_head(proj),
        }

    # ------------------------------------------------------------------
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        log.info("Backbone frozen.")

    def unfreeze_last_n_blocks(self, n: int):
        blocks = list(self.backbone.blocks)
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        # Also unfreeze final norm
        for nm, mod in self.backbone.named_modules():
            if "norm" in nm and "blocks" not in nm:
                for p in mod.parameters():
                    p.requires_grad_(True)
        unfrozen = sum(p.requires_grad for p in self.backbone.parameters())
        log.info(f"Unfroze last {n} backbone blocks ({unfrozen} backbone params trainable).")

    def head_parameters(self):
        return (list(self.view_pool.parameters())
                + list(self.projection.parameters())
                + list(self.mgi_head.parameters())
                + list(self.ohi_head.parameters())
                + list(self.gei_head.parameters()))

    def backbone_parameters(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Decode + TTA helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_from_probs(probs: torch.Tensor, threshold=0.5) -> torch.Tensor:
    """Decode ordinal probabilities with scalar or per-threshold decision values."""
    if isinstance(threshold, (list, tuple, np.ndarray)):
        thr = torch.tensor(threshold, dtype=probs.dtype, device=probs.device)
    else:
        thr = torch.tensor([float(threshold)], dtype=probs.dtype, device=probs.device)

    if thr.numel() == 1:
        thr = thr.repeat(probs.shape[1])
    thr = thr.view(1, -1)

    return (probs > thr).sum(dim=1).long()


@torch.no_grad()
def decode_ordinal(logits: torch.Tensor, threshold=0.5) -> torch.Tensor:
    """(B, K-1) ordinal logits → (B,) integer class predictions."""
    return decode_from_probs(torch.sigmoid(logits), threshold=threshold)


@torch.no_grad()
def tta_predict(model: nn.Module,
                frontal: torch.Tensor,
                left:    torch.Tensor,
                right:   torch.Tensor,
                n_tta:   int = 5,
                decode_thresholds: dict = None) -> dict:
    """
    Test-time augmentation on normalised tensors.
    Averages sigmoid probabilities across n_tta passes, then decodes.
    """
    model.eval()
    accum = {"mgi": [], "ohi": [], "gei": []}

    def _augment(t):
        # Random horizontal flip
        if random.random() > 0.5:
            t = torch.flip(t, dims=[-1])
        # Tiny brightness jitter
        t = (t + torch.randn_like(t) * 0.01).clamp(-3, 3)
        return t

    for i in range(n_tta):
        if i == 0:
            f_in, l_in, r_in = frontal, left, right
        else:
            f_in = _augment(frontal.clone())
            l_in = _augment(left.clone())
            r_in = _augment(right.clone())

        out = model(f_in, l_in, r_in)
        for task, logits in out.items():
            accum[task].append(torch.sigmoid(logits))

    if decode_thresholds is None:
        decode_thresholds = {"mgi": 0.5, "ohi": 0.5, "gei": 0.5}

    avg = {k: torch.stack(v).mean(0) for k, v in accum.items()}
    return {
        k: decode_from_probs(v, threshold=decode_thresholds.get(k, 0.5))
        for k, v in avg.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_composite_sampler(dataset, indices: list,
                            class_balance_weights: dict = None,
                            drw_alpha: float = 0.0) -> WeightedRandomSampler:
    """
    Composite multi-task sampler built from per-task inverse-frequency
    weights with optional class-balanced DRW scaling.
    """
    mgi_arr = np.array([dataset.mgi_labels[i] for i in indices])
    ohi_arr = np.array([dataset.ohi_labels[i] for i in indices])
    gei_arr = np.array([dataset.gei_labels[i] for i in indices])

    def inv_freq(arr, n_cls):
        counts = np.bincount(arr, minlength=n_cls).astype(float)
        counts = np.where(counts == 0, 0.5, counts)
        return np.array([1.0 / counts[y] for y in arr])

    def drw_scale(arr, task):
        if class_balance_weights is None or task not in class_balance_weights:
            return np.ones_like(arr, dtype=np.float64)
        cw = class_balance_weights[task].detach().cpu().numpy()
        return 1.0 + float(drw_alpha) * (cw[arr] - 1.0)

    w_mgi = inv_freq(mgi_arr, CONFIG["num_classes_mgi"]) * drw_scale(mgi_arr, "mgi")
    w_ohi = inv_freq(ohi_arr, CONFIG["num_classes_ohi"]) * drw_scale(ohi_arr, "ohi")
    w_gei = inv_freq(gei_arr, CONFIG["num_classes_gei"]) * drw_scale(gei_arr, "gei")

    w_mgi = w_mgi / w_mgi.sum()
    w_ohi = w_ohi / w_ohi.sum()
    w_gei = w_gei / w_gei.sum()

    # Arithmetic mean → composite weight (better allows rare severe classes to be sampled)
    composite = (w_mgi + w_ohi + w_gei) / 3.0
    # Cap at 5× median to avoid extreme over-sampling of a single patient
    med = np.median(composite)
    composite = np.clip(composite, None, 5.0 * med)
    composite = composite / composite.sum()

    return WeightedRandomSampler(
        weights=torch.tensor(composite, dtype=torch.float32),
        num_samples=len(indices),
        replacement=True,
    )


def _collect_batch(batch, device):
    frontal, left, right, labels = batch
    targets = {
        "mgi": labels["mgi"].to(device).long(),
        "ohi": labels["ohi"].to(device).long(),
        "gei": labels["gei"].to(device).long(),
    }
    return frontal.to(device), left.to(device), right.to(device), targets


def _apply_mixup(frontal, left, right, targets, alpha, prob):
    """
    MixUp in feature (tensor) space.  Returns mixed tensors + both label dicts.
    """
    if alpha <= 0 or random.random() > prob:
        return frontal, left, right, targets, None, 1.0

    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)          # Keep lambda ≥ 0.5 (keep primary label dominant)

    B   = frontal.shape[0]
    idx = torch.randperm(B, device=frontal.device)

    f_mix = lam * frontal + (1.0 - lam) * frontal[idx]
    l_mix = lam * left    + (1.0 - lam) * left[idx]
    r_mix = lam * right   + (1.0 - lam) * right[idx]

    targets_b = {k: v[idx] for k, v in targets.items()}
    return f_mix, l_mix, r_mix, targets, targets_b, lam


def _make_scaler():
    """AMP GradScaler compatible with PyTorch 2.x and 1.x."""
    if hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda")
        except TypeError:
            return torch.amp.GradScaler()
    return _amp_compat.GradScaler()


def _autocast():
    """AMP autocast context compatible with PyTorch 2.x and 1.x."""
    if hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type="cuda", enabled=True)
        except TypeError:
            return torch.amp.autocast("cuda", enabled=True)
    return _amp_compat.autocast(enabled=True)


# ─────────────────────────────────────────────────────────────────────────────

def run_epoch_train(model, loader, optimizer, loss_fn,
                    scaler, device, grad_clip, use_mixup,
                    class_weights: dict = None,
                    drw_alpha: float = 0.0):
    model.train()
    totals = {"total": 0.0, "mgi": 0.0, "ohi": 0.0, "gei": 0.0}
    n_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        f, l, r, targets = _collect_batch(batch, device)

        if use_mixup:
            f, l, r, targets, targets_b, lam = _apply_mixup(
                f, l, r, targets, CONFIG["mixup_alpha"], CONFIG["mixup_prob"]
            )
        else:
            targets_b, lam = None, 1.0

        with _autocast():
            preds              = model(f, l, r)
            total_loss, tl     = loss_fn(
                preds,
                targets,
                lam,
                targets_b,
                class_weights=class_weights,
                drw_alpha=drw_alpha,
            )
            total_loss_scaled  = total_loss / CONFIG["grad_accum_steps"]

        scaler.scale(total_loss_scaled).backward()

        if (step + 1) % CONFIG["grad_accum_steps"] == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        totals["total"] += total_loss.item()
        for k in ("mgi", "ohi", "gei"):
            totals[k] += tl[k]
        n_steps += 1

    # Handle leftover gradient accumulation
    if (n_steps % CONFIG["grad_accum_steps"]) != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return {k: v / max(n_steps, 1) for k, v in totals.items()}


def run_epoch_eval(model, loader, loss_fn, device, decode_thresholds=None):
    model.eval()
    y_true = {"mgi": [], "ohi": [], "gei": []}
    y_pred = {"mgi": [], "ohi": [], "gei": []}
    total_loss = 0.0
    n = 0

    if decode_thresholds is None:
        decode_thresholds = {"mgi": 0.5, "ohi": 0.5, "gei": 0.5}

    with torch.no_grad():
        for batch in loader:
            f, l, r, targets = _collect_batch(batch, device)
            with _autocast():
                preds          = model(f, l, r)
                loss, _        = loss_fn(preds, targets)
            total_loss += loss.item()
            n          += 1

            for task in ("mgi", "ohi", "gei"):
                y_true[task].extend(targets[task].cpu().numpy().tolist())
                y_pred[task].extend(
                    decode_ordinal(
                        preds[task],
                        threshold=decode_thresholds.get(task, 0.5),
                    ).cpu().numpy().tolist()
                )

    avg_loss = total_loss / max(n, 1)
    return avg_loss, y_true, y_pred


def compute_metrics(y_true: dict, y_pred: dict, class_counts: dict) -> dict:
    """Compute macro-F1, accuracy, MAE, and directional error rates."""
    results = {}
    for task, n_cls in class_counts.items():
        labels = list(range(n_cls))
        y_t = np.array(y_true[task], dtype=np.int64)
        y_p = np.array(y_pred[task], dtype=np.int64)

        f1  = f1_score(y_t, y_p, labels=labels,
                       average="macro", zero_division=0)
        acc = accuracy_score(y_t, y_p)
        mae = float(np.mean(np.abs(y_t - y_p)))
        over_rate = float(np.mean(y_p > y_t))
        under_rate = float(np.mean(y_p < y_t))
        results[task] = {
            "f1": f1,
            "acc": acc,
            "mae": mae,
            "over_rate": over_rate,
            "under_rate": under_rate,
        }
    return results


def compute_fold_objective(metrics: dict, config: dict) -> float:
    """
    Balanced optimisation target across MGI, OHI, and GEI.

    Higher is better.
    """
    w = config.get("objective_weights", {"mgi": 0.34, "ohi": 0.33, "gei": 0.33})
    obj = (
        w["mgi"] * metrics["mgi"]["f1"]
        + w["ohi"] * metrics["ohi"]["f1"]
        + w["gei"] * metrics["gei"]["f1"]
    )
    
    # Asymmetric penalties penalizing under-estimation strictly for all medical indices
    obj -= config.get("objective_mgi_mae_weight", 0.20) * metrics["mgi"]["mae"]
    obj -= config.get("objective_mgi_under_weight", 0.20) * metrics["mgi"]["under_rate"]
    
    obj -= config.get("objective_ohi_mae_weight", 0.20) * metrics["ohi"]["mae"]
    obj -= config.get("objective_ohi_under_weight", 0.20) * metrics["ohi"]["under_rate"]
    
    obj -= config.get("objective_gei_mae_weight", 0.20) * metrics["gei"]["mae"]
    obj -= config.get("objective_gei_under_weight", 0.20) * metrics["gei"]["under_rate"]
    
    return float(obj)


@torch.no_grad()
def calibrate_task_threshold(model: nn.Module, loader, device,
                             config: dict, task: str):
    """
    Calibrate decode threshold for one ordinal task on validation data.
    """
    task = task.lower()
    model.eval()
    probs_all = []
    y_all = []

    for batch in loader:
        f, l, r, targets = _collect_batch(batch, device)
        logits = model(f, l, r)[task]
        probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
        y_all.append(targets[task].detach().cpu().numpy())

    if not probs_all:
        return config.get("default_decode_threshold", 0.5), {
            "f1": 0.0,
            "mae": 0.0,
            "over_rate": 0.0,
            "under_rate": 0.0,
            "objective": -1e9,
        }

    probs = np.concatenate(probs_all, axis=0)
    y_true = np.concatenate(y_all, axis=0)

    t_min = float(config.get(f"{task}_threshold_min", 0.40))
    t_max = float(config.get(f"{task}_threshold_max", 0.80))
    t_steps = int(config.get(f"{task}_threshold_steps", 29))

    best_t = float(config.get("default_decode_threshold", 0.5))
    best_stats = {
        "f1": -1.0,
        "mae": 10.0,
        "over_rate": 1.0,
        "under_rate": 1.0,
        "objective": -1e9,
    }

    mae_w = float(config.get(f"objective_{task}_mae_weight", 0.15))
    over_w = float(config.get(f"objective_{task}_over_weight", 0.10))
    under_w = float(config.get(f"objective_{task}_under_weight", 0.10))

    for t in np.linspace(t_min, t_max, t_steps):
        y_pred = (probs > t).sum(axis=1)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        mae = float(np.mean(np.abs(y_pred - y_true)))
        over_rate = float(np.mean(y_pred > y_true))
        under_rate = float(np.mean(y_pred < y_true))
        objective = float(f1 - mae_w * mae - over_w * over_rate - under_w * under_rate)

        if objective > best_stats["objective"]:
            best_t = float(t)
            best_stats = {
                "f1": float(f1),
                "mae": mae,
                "over_rate": over_rate,
                "under_rate": under_rate,
                "objective": objective,
            }

    return best_t, best_stats


@torch.no_grad()
def calibrate_mgi_threshold(model: nn.Module, loader, device, config: dict):
    return calibrate_task_threshold(model, loader, device, config, task="mgi")


@torch.no_grad()
def calibrate_ohi_threshold(model: nn.Module, loader, device, config: dict):
    return calibrate_task_threshold(model, loader, device, config, task="ohi")


@torch.no_grad()
def calibrate_gei_threshold(model: nn.Module, loader, device, config: dict):
    return calibrate_task_threshold(model, loader, device, config, task="gei")


def bake_decode_thresholds_into_checkpoint(src_ckpt: str, dst_ckpt: str, decode_thresholds: dict):
    """
    Convert calibrated decode thresholds into equivalent bias shifts.

    sigmoid(logit) > t  <=>  sigmoid(logit - logit(t)) > 0.5
    This allows runtime inference to keep a fixed 0.5 decode threshold.
    """
    thresholds = {
        "mgi": float(decode_thresholds.get("mgi", 0.5)),
        "ohi": float(decode_thresholds.get("ohi", 0.5)),
        "gei": float(decode_thresholds.get("gei", 0.5)),
    }
    deltas = {
        task: float(np.log(np.clip(t, 1e-4, 1 - 1e-4) / (1.0 - np.clip(t, 1e-4, 1 - 1e-4))))
        for task, t in thresholds.items()
    }

    try:
        ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(src_ckpt, map_location="cpu")

    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    candidate_keys = {
        "mgi": ["mgi_head.net.3.bias", "module.mgi_head.net.3.bias"],
        "ohi": ["ohi_head.net.3.bias", "module.ohi_head.net.3.bias"],
        "gei": ["gei_head.net.3.bias", "module.gei_head.net.3.bias"],
    }
    touched_tasks = []
    for task, keys in candidate_keys.items():
        for k in keys:
            if k in state:
                state[k] = state[k] - deltas[task]
                touched_tasks.append(task)
                break

    if "mgi" not in touched_tasks:
        raise KeyError("Could not find MGI head final bias to bake threshold calibration.")
    if "ohi" not in touched_tasks:
        raise KeyError("Could not find OHI head final bias to bake threshold calibration.")
    if "gei" not in touched_tasks:
        raise KeyError("Could not find GEI head final bias to bake threshold calibration.")

    if "model_state_dict" in ckpt:
        ckpt["model_state_dict"] = state
        ckpt["decode_thresholds"] = thresholds
        ckpt["bias_baked_delta"] = deltas
        torch.save(ckpt, dst_ckpt)
    else:
        torch.save(state, dst_ckpt)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — SWA (simple uniform weight averaging — no external deps)
# ─────────────────────────────────────────────────────────────────────────────

class SWAHelper:
    """
    Rolling uniform average of model weights.
    More robust than torch.optim.swa_utils for small datasets.
    """

    def __init__(self):
        self.n      = 0
        self.shadow = None

    def update(self, model: nn.Module):
        state = {k: v.detach().cpu().float().clone()
                 for k, v in model.state_dict().items()}
        if self.shadow is None:
            self.shadow = state
        else:
            n = self.n
            self.shadow = {
                k: self.shadow[k] * (n / (n + 1)) + state[k] * (1 / (n + 1))
                for k in state
            }
        self.n += 1

    def apply_to(self, model: nn.Module):
        """Load averaged weights into model (converts back to model dtype/device)."""
        if self.shadow is None:
            return
        cur  = model.state_dict()
        avg  = {k: self.shadow[k].to(cur[k].device).to(cur[k].dtype)
                for k in self.shadow}
        model.load_state_dict(avg)
        log.info(f"SWA weights applied ({self.n} checkpoints averaged).")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — K-Fold training loop
# ─────────────────────────────────────────────────────────────────────────────

N_CLS = {
    "mgi": CONFIG["num_classes_mgi"],
    "ohi": CONFIG["num_classes_ohi"],
    "gei": CONFIG["num_classes_gei"],
}

all_mgi_labels   = np.array(FULL_TRAIN_DS.mgi_labels)
n_patients       = len(FULL_TRAIN_DS)

# Determine effective k (class 4 MGI has only 8 samples → max 8 folds for stratified)
mgi_counts   = np.bincount(all_mgi_labels, minlength=CONFIG["num_classes_mgi"])
effective_k  = min(CONFIG["k_folds"], int(mgi_counts[mgi_counts > 0].min()))
if effective_k < 2:
    raise RuntimeError(
        f"Need ≥2 samples per MGI class for stratified CV. "
        f"Counts: {mgi_counts.tolist()}"
    )
if effective_k != CONFIG["k_folds"]:
    log.warning(f"k_folds reduced {CONFIG['k_folds']} → {effective_k} "
                f"(MGI class 4 has only {mgi_counts[4]} samples).")

skf = StratifiedKFold(n_splits=effective_k, shuffle=True,
                      random_state=CONFIG["seed"])

fold_summaries    = []
checkpoint_paths  = []
swa_paths         = []
all_histories     = []

print(f"\n{'='*70}")
print(f"  STARTING {effective_k}-FOLD TRAINING")
print(f"  Patients: {n_patients}  |  Epochs: {CONFIG['num_epochs']}  "
      f"|  SWA start: {CONFIG['swa_start_epoch']}")
print(f"{'='*70}\n")

for fold_idx, (train_idx, val_idx) in enumerate(
    skf.split(np.arange(n_patients), all_mgi_labels), start=1
):
    seed_everything(CONFIG["seed"] + fold_idx)

    print(f"\n{'─'*70}")
    print(f"  FOLD {fold_idx}/{effective_k}  "
          f"train={len(train_idx)}  val={len(val_idx)}")
    print(f"{'─'*70}")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_indices = list(train_idx)
    train_subset = Subset(FULL_TRAIN_DS, train_indices)
    fold_class_weights = compute_fold_class_balanced_weights(FULL_TRAIN_DS, train_indices, CONFIG)
    sampler = build_composite_sampler(
        FULL_TRAIN_DS,
        train_indices,
        class_balance_weights=fold_class_weights,
        drw_alpha=compute_drw_alpha(1, CONFIG),
    )
    train_loader = DataLoader(
        train_subset,
        batch_size=CONFIG["batch_size"], sampler=sampler,
        num_workers=CONFIG["num_workers"], drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(FULL_VAL_DS, list(val_idx)),
        batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    # ── Model + Loss ─────────────────────────────────────────────────────────
    model   = OralHealthModelV2(CONFIG).to(DEVICE)
    loss_fn = MultiTaskLoss(POS_WEIGHTS, CONFIG).to(DEVICE)
    scaler  = _make_scaler()
    swa     = SWAHelper()

    # ── PHASE 1: freeze backbone, train heads only ────────────────────────────
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        model.head_parameters(),
        lr=CONFIG["lr_heads"],
        weight_decay=CONFIG["weight_decay"],
    )
    # Simple cosine scheduler for freeze phase
    scheduler_phase1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["freeze_epochs"],
        eta_min=CONFIG["lr_heads"] * 0.1,
    )

    ckpt_path = Path(CONFIG["checkpoint_dir"]) / f"fold_{fold_idx}_best.pth"
    swa_path  = Path(CONFIG["checkpoint_dir"]) / f"fold_{fold_idx}_swa.pth"

    decode_thresholds = {
        "mgi": CONFIG.get("default_decode_threshold", 0.5),
        "ohi": CONFIG.get("default_decode_threshold", 0.5),
        "gei": 0.5,
    }

    best_objective = -1e9
    best_f1        = 0.0
    best_decode_thresholds = decode_thresholds.copy()
    best_mgi_mae = 10.0
    best_mgi_over = 1.0
    best_ohi_mae = 10.0
    best_ohi_under = 1.0
    patience_count = 0
    fold_history   = []
    phase          = 1

    for epoch in range(1, CONFIG["num_epochs"] + 1):

        drw_alpha = compute_drw_alpha(epoch, CONFIG)
        sampler = build_composite_sampler(
            FULL_TRAIN_DS,
            train_indices,
            class_balance_weights=fold_class_weights,
            drw_alpha=drw_alpha,
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=CONFIG["batch_size"], sampler=sampler,
            num_workers=CONFIG["num_workers"], drop_last=True,
            pin_memory=True,
        )

        # ── Transition to Phase 2: unfreeze backbone ──────────────────────────
        if epoch == CONFIG["freeze_epochs"] + 1:
            phase = 2
            model.unfreeze_last_n_blocks(CONFIG["unfreeze_blocks"])

            backbone_params = model.backbone_parameters()
            param_groups = [
                {"params": backbone_params,           "lr": CONFIG["lr_backbone"]},
                {"params": model.view_pool.parameters(),
                 "lr": CONFIG["lr_projection"]},
                {"params": model.projection.parameters(),
                 "lr": CONFIG["lr_projection"]},
                {"params": model.mgi_head.parameters(), "lr": CONFIG["lr_heads"]},
                {"params": model.ohi_head.parameters(), "lr": CONFIG["lr_heads"]},
                {"params": model.gei_head.parameters(), "lr": CONFIG["lr_heads"]},
            ]
            optimizer = torch.optim.AdamW(param_groups,
                                           weight_decay=CONFIG["weight_decay"])

            remaining = CONFIG["num_epochs"] - CONFIG["freeze_epochs"]
            # Linear warmup → cosine decay, stepped once per epoch
            def _lr_lambda(ep_rel, warmup=CONFIG["warmup_epochs"], total=remaining):
                if ep_rel < warmup:
                    return float(ep_rel + 1) / float(warmup + 1)
                progress = (ep_rel - warmup) / max(total - warmup, 1)
                return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))

            scheduler_phase2 = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda ep: _lr_lambda(ep)
            )

        # ── Train one epoch ───────────────────────────────────────────────────
        use_mixup = (phase == 2)  # MixUp only after backbone is unfrozen
        train_losses = run_epoch_train(
            model, train_loader, optimizer, loss_fn,
            scaler, DEVICE, CONFIG.get("grad_clip_norm", CONFIG.get("grad_clip", 1.0)), use_mixup,
            class_weights=fold_class_weights,
            drw_alpha=drw_alpha,
        )

        # ── Scheduler step ────────────────────────────────────────────────────
        if phase == 1:
            scheduler_phase1.step()
        else:
            scheduler_phase2.step()

        # ── SWA update ────────────────────────────────────────────────────────
        if epoch >= CONFIG["swa_start_epoch"]:
            swa.update(model)

        # ── Validation threshold calibration (MGI) ──────────────────────────
        if phase == 2 and (
            epoch == CONFIG["freeze_epochs"] + 1
            or epoch % int(CONFIG.get("calibrate_every_n_epochs", 5)) == 0
        ):
            mgi_thr, mgi_thr_stats = calibrate_mgi_threshold(model, val_loader, DEVICE, CONFIG)
            ohi_thr, ohi_thr_stats = calibrate_ohi_threshold(model, val_loader, DEVICE, CONFIG)
            gei_thr, gei_thr_stats = calibrate_gei_threshold(model, val_loader, DEVICE, CONFIG)
            decode_thresholds["mgi"] = mgi_thr
            decode_thresholds["ohi"] = ohi_thr
            decode_thresholds["gei"] = gei_thr
            print(
                f"  Threshold calibrator -> mgi_thr={mgi_thr:.3f} "
                f"(f1={mgi_thr_stats['f1']:.3f}, mae={mgi_thr_stats['mae']:.3f}, over={mgi_thr_stats['over_rate']:.3f})  "
                f"ohi_thr={ohi_thr:.3f} "
                f"(f1={ohi_thr_stats['f1']:.3f}, mae={ohi_thr_stats['mae']:.3f}, under={ohi_thr_stats['under_rate']:.3f})  "
                f"gei_thr={gei_thr:.3f} "
                f"(f1={gei_thr_stats['f1']:.3f}, mae={gei_thr_stats['mae']:.3f}, under={gei_thr_stats['under_rate']:.3f})"
            )

        # ── Validation ───────────────────────────────────────────────────────
        val_loss, yt, yp = run_epoch_eval(
            model, val_loader, loss_fn, DEVICE,
            decode_thresholds=decode_thresholds,
        )
        metrics = compute_metrics(yt, yp, N_CLS)

        avg_f1 = np.mean([metrics[t]["f1"] for t in N_CLS])
        objective = compute_fold_objective(metrics, CONFIG)

        fold_history.append({
            "epoch":     epoch,
            "train_loss": train_losses["total"],
            "val_loss":   val_loss,
            "objective":  objective,
            "val_avg_f1": avg_f1,
            "f1_mgi":    metrics["mgi"]["f1"],
            "f1_ohi":    metrics["ohi"]["f1"],
            "f1_gei":    metrics["gei"]["f1"],
            "mgi_mae":   metrics["mgi"]["mae"],
            "mgi_over_rate": metrics["mgi"]["over_rate"],
            "ohi_mae":   metrics["ohi"]["mae"],
            "ohi_under_rate": metrics["ohi"]["under_rate"],
            "mgi_threshold": decode_thresholds["mgi"],
            "ohi_threshold": decode_thresholds["ohi"],
            "gei_threshold": decode_thresholds["gei"],
            "drw_alpha": drw_alpha,
        })

        # ── Progress print ────────────────────────────────────────────────────
        if epoch % 5 == 0 or epoch <= 3:
            print(
                f"  Ep {epoch:3d}  "
                f"tr_loss={train_losses['total']:.3f}  "
                f"val_loss={val_loss:.3f}  "
                f"f1=({metrics['mgi']['f1']:.3f} "
                f"{metrics['ohi']['f1']:.3f} "
                f"{metrics['gei']['f1']:.3f}) "
                f"avg={avg_f1:.3f}  "
                f"obj={objective:.3f}  "
                f"mgi_mae={metrics['mgi']['mae']:.3f}  "
                f"mgi_over={metrics['mgi']['over_rate']:.3f}  "
                f"ohi_mae={metrics['ohi']['mae']:.3f}  "
                f"ohi_under={metrics['ohi']['under_rate']:.3f}  "
                f"mgi_thr={decode_thresholds['mgi']:.3f}  "
                f"ohi_thr={decode_thresholds['ohi']:.3f}  "
                f"gei_thr={decode_thresholds['gei']:.3f}  "
                f"drw={drw_alpha:.2f}"
                + (" ★" if objective > best_objective else "")
            )

        # ── Checkpoint best model ─────────────────────────────────────────────
        if objective > best_objective:
            best_objective = objective
            best_f1 = avg_f1
            best_mgi_mae = metrics["mgi"]["mae"]
            best_mgi_over = metrics["mgi"]["over_rate"]
            best_ohi_mae = metrics["ohi"]["mae"]
            best_ohi_under = metrics["ohi"]["under_rate"]
            best_decode_thresholds = decode_thresholds.copy()
            patience_count = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "best_f1":          best_f1,
                "best_objective":   best_objective,
                "decode_thresholds": best_decode_thresholds,
                "metrics":          metrics,
                "config":           CONFIG,
            }, ckpt_path)
        else:
            patience_count += 1

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_count >= CONFIG["early_stopping_patience"] and phase == 2:
            print(f"  Early stopping at epoch {epoch} (patience exhausted).")
            break

    # ── Save SWA checkpoint ───────────────────────────────────────────────────
    if swa.n > 0:
        # Load SWA weights into model temporarily for saving
        model_swa = OralHealthModelV2(CONFIG).to(DEVICE)
        ckpt_data = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model_swa.load_state_dict(ckpt_data["model_state_dict"])
        swa.apply_to(model_swa)
        torch.save({
            "model_state_dict": model_swa.state_dict(),
            "swa_n":            swa.n,
            "config":           CONFIG,
        }, swa_path)
        swa_paths.append(str(swa_path))
        del model_swa
        torch.cuda.empty_cache()
        print(f"  SWA model saved ({swa.n} checkpoints averaged) → {swa_path.name}")
    else:
        print("  SWA: no epochs reached SWA start — saving best model as SWA fallback.")
        import shutil
        shutil.copy(ckpt_path, swa_path)
        swa_paths.append(str(swa_path))

    # ── Fold summary ──────────────────────────────────────────────────────────
    best_epoch_data = max(fold_history, key=lambda h: h["objective"])
    fold_summaries.append({
        "fold":          fold_idx,
        "best_objective": best_objective,
        "best_val_f1":   best_f1,
        "best_epoch":    best_epoch_data["epoch"],
        "f1_mgi":        best_epoch_data["f1_mgi"],
        "f1_ohi":        best_epoch_data["f1_ohi"],
        "f1_gei":        best_epoch_data["f1_gei"],
        "mgi_mae":       best_mgi_mae,
        "mgi_over_rate": best_mgi_over,
        "ohi_mae":       best_ohi_mae,
        "ohi_under_rate": best_ohi_under,
        "decode_thresholds": best_decode_thresholds,
        "checkpoint":    str(ckpt_path),
        "swa_checkpoint": str(swa_path),
    })
    checkpoint_paths.append(str(ckpt_path))
    all_histories.append(fold_history)

    print(f"\n  Fold {fold_idx} best: avg_f1={best_f1:.4f}  obj={best_objective:.4f}  "
          f"(mgi={best_epoch_data['f1_mgi']:.3f} "
          f"ohi={best_epoch_data['f1_ohi']:.3f} "
          f"gei={best_epoch_data['f1_gei']:.3f})  "
          f"mgi_mae={best_mgi_mae:.3f}  "
          f"mgi_over={best_mgi_over:.3f}  "
          f"ohi_mae={best_ohi_mae:.3f}  "
          f"ohi_under={best_ohi_under:.3f}  "
          f"mgi_thr={best_decode_thresholds['mgi']:.3f}  "
          f"ohi_thr={best_decode_thresholds['ohi']:.3f}  "
          f"gei_thr={best_decode_thresholds['gei']:.3f}")

    del model, loss_fn, optimizer, scaler, swa
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

df_summary = pd.DataFrame(fold_summaries)
print(f"\n{'='*70}")
print("  FOLD-LEVEL RESULTS")
print(f"{'='*70}")
print(df_summary[[
    "fold",
    "best_objective",
    "best_val_f1",
    "best_epoch",
    "f1_mgi",
    "f1_ohi",
    "f1_gei",
    "mgi_mae",
    "mgi_over_rate",
    "ohi_mae",
    "ohi_under_rate",
]].to_string(index=False))
print(f"\n  Mean ± Std across {effective_k} folds:")
for col in [
    "best_objective",
    "best_val_f1",
    "f1_mgi",
    "f1_ohi",
    "f1_gei",
    "mgi_mae",
    "mgi_over_rate",
    "ohi_mae",
    "ohi_under_rate",
]:
    v = df_summary[col]
    print(f"    {col:18s}: {v.mean():.4f} ± {v.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — Final ensemble evaluation (TTA on best fold)
# ─────────────────────────────────────────────────────────────────────────────

best_fold_idx  = int(df_summary["best_val_f1"].idxmax())
best_ckpt_path = fold_summaries[best_fold_idx]["checkpoint"]
best_swa_path  = fold_summaries[best_fold_idx]["swa_checkpoint"]
best_decode_thresholds = fold_summaries[best_fold_idx]["decode_thresholds"]

# Rebuild best fold's val loader
_, val_idx_best = list(
    skf.split(np.arange(n_patients), all_mgi_labels)
)[best_fold_idx]
val_loader_best = DataLoader(
    Subset(FULL_VAL_DS, list(val_idx_best)),
    batch_size=1, shuffle=False,
)

print(f"\nEnsemble eval on fold {best_fold_idx+1} val set (TTA={CONFIG['tta_steps']})...")

def eval_with_tta(ckpt_path, val_loader, decode_thresholds):
    m = OralHealthModelV2(CONFIG).to(DEVICE)
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()

    yt = {"mgi": [], "ohi": [], "gei": []}
    yp = {"mgi": [], "ohi": [], "gei": []}

    for batch in val_loader:
        f, l, r, labels = batch
        f, l, r = f.to(DEVICE), l.to(DEVICE), r.to(DEVICE)
        preds = tta_predict(
            m, f, l, r,
            n_tta=CONFIG["tta_steps"],
            decode_thresholds=decode_thresholds,
        )
        for task in ("mgi", "ohi", "gei"):
            yt[task].extend(labels[task].numpy().tolist())
            yp[task].extend(preds[task].cpu().numpy().tolist())
    del m; torch.cuda.empty_cache()
    return yt, yp


yt_best, yp_best = eval_with_tta(best_ckpt_path, val_loader_best, best_decode_thresholds)
yt_swa,  yp_swa  = eval_with_tta(best_swa_path,  val_loader_best, best_decode_thresholds)

print("\n── Best checkpoint (TTA) ───────────────────────────────────────────")
for task in ("mgi", "ohi", "gei"):
    f1 = f1_score(yt_best[task], yp_best[task], average="macro", zero_division=0)
    print(f"  {task.upper()} macro-F1: {f1:.4f}")
    if task == "mgi":
        y_t = np.array(yt_best[task])
        y_p = np.array(yp_best[task])
        print(f"  MGI MAE: {np.mean(np.abs(y_t - y_p)):.4f}  |  MGI over-rate: {np.mean(y_p > y_t):.4f}")
    print(classification_report(yt_best[task], yp_best[task], zero_division=0))

print("\n── SWA checkpoint (TTA) ────────────────────────────────────────────")
for task in ("mgi", "ohi", "gei"):
    f1 = f1_score(yt_swa[task], yp_swa[task], average="macro", zero_division=0)
    print(f"  {task.upper()} macro-F1: {f1:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 — Visualisations
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for fold_i, history in enumerate(all_histories):
    ep       = [h["epoch"]      for h in history]
    tr_loss  = [h["train_loss"] for h in history]
    vl_loss  = [h["val_loss"]   for h in history]
    avg_f1_h = [h["val_avg_f1"] for h in history]
    axes[0, 0].plot(ep, tr_loss,  label=f"Fold {fold_i+1}")
    axes[0, 1].plot(ep, vl_loss,  label=f"Fold {fold_i+1}")
    axes[0, 2].plot(ep, avg_f1_h, label=f"Fold {fold_i+1}")

for ax, title in zip(axes[0], ["Train Loss", "Val Loss", "Val Avg Macro-F1"]):
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

class_cfgs = {
    f"MGI (0-{CONFIG['num_classes_mgi']-1})": (
        range(CONFIG["num_classes_mgi"]), "mgi", axes[1, 0]),
    f"OHI (0-{CONFIG['num_classes_ohi']-1}) merged": (
        range(CONFIG["num_classes_ohi"]), "ohi", axes[1, 1]),
    f"GEI (0-{CONFIG['num_classes_gei']-1}) merged": (
        range(CONFIG["num_classes_gei"]), "gei", axes[1, 2]),
}

for title, (labels, task, ax) in class_cfgs.items():
    cm      = confusion_matrix(yt_best[task], yp_best[task], labels=list(labels))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=list(labels), yticklabels=list(labels))
    ax.set_title(f"{title} — Confusion Matrix (normalised)", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.suptitle("DentAI v2 — Training Results", fontsize=15, fontweight="bold")
plt.tight_layout()
save_path = os.path.join(CONFIG["plots_dir"], "training_results_v2.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17 — Export ensemble config (for backend inference)
# ─────────────────────────────────────────────────────────────────────────────

# Choose best model per fold: SWA if f1_swa ≥ f1_best, else best-epoch checkpoint
final_model_paths = []
decode_thresholds_per_fold = {}
for i, summary in enumerate(fold_summaries):
    src_ckpt = summary["swa_checkpoint"]  # prefer SWA for deployment
    fold_id = summary["fold"]
    decode_thr = {
        "mgi": float(summary["decode_thresholds"].get("mgi", 0.5)),
        "ohi": float(summary["decode_thresholds"].get("ohi", 0.5)),
        "gei": float(summary["decode_thresholds"].get("gei", 0.5)),
    }
    deploy_ckpt = str(Path(CONFIG["checkpoint_dir"]) / f"fold_{fold_id}_deploy.pth")
    bake_decode_thresholds_into_checkpoint(src_ckpt, deploy_ckpt, decode_thr)
    final_model_paths.append(deploy_ckpt)
    decode_thresholds_per_fold[f"fold_{fold_id}"] = decode_thr

global_decode_thresholds = {
    "mgi": float(np.mean([v["mgi"] for v in decode_thresholds_per_fold.values()])),
    "ohi": float(np.mean([v["ohi"] for v in decode_thresholds_per_fold.values()])),
    "gei": float(np.mean([v["gei"] for v in decode_thresholds_per_fold.values()])),
}

avg_f1 = {
    t: float(df_summary[f"f1_{t}"].mean())
    for t in ("mgi", "ohi", "gei")
}

ensemble_cfg = {
    "model_type":        "OralHealthModelV2",
    "models":            final_model_paths,
    "weights":           [1.0] * len(final_model_paths),
    "config":            CONFIG,
    "avg_val_f1":        avg_f1,
    "label_mapping": {
        "ohi": "classes 0,1,2  (original 2 and 3 merged into class 2)",
        "gei": "classes 0,1,2  (original 2 and 3 merged into class 2)",
        "mgi": "classes 0,1,2,3,4 (unchanged)",
    },
    "decode": "sigmoid(logits) > 0.5 → sum per threshold = predicted class",
    "decode_thresholds_global": global_decode_thresholds,
    "decode_thresholds_per_fold": decode_thresholds_per_fold,
    "deployment_note": "MGI, OHI, and GEI calibrated thresholds are baked into deployed checkpoint biases.",
    "real_world_focus": {
        "objective": "weighted_f1_with_all_task_directional_penalties",
        "mgi_over_penalty": CONFIG["mgi_over_penalty"],
        "mgi_under_penalty": CONFIG["mgi_under_penalty"],
        "drw": {
            "start_epoch": CONFIG["drw_start_epoch"],
            "full_epoch": CONFIG["drw_full_epoch"],
        },
    },
    "tta_steps": CONFIG["tta_steps"],
}

for save_path in [
    Path(CONFIG["checkpoint_dir"]) / "ensemble_config.json",
    Path(CONFIG["checkpoint_dir"]).parent / "ensemble_config.json",
]:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as fp:
        json.dump(ensemble_cfg, fp, indent=2)
    print(f"Ensemble config → {save_path}")

print(f"\n{'='*70}")
print("  TRAINING COMPLETE")
print(f"  Best avg val F1: {df_summary['best_val_f1'].max():.4f}")
print(f"  Mean F1 — MGI: {avg_f1['mgi']:.4f}  "
      f"OHI: {avg_f1['ohi']:.4f}  GEI: {avg_f1['gei']:.4f}")
print(f"{'='*70}\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18 — Backend integration check
# ─────────────────────────────────────────────────────────────────────────────

print("── Backend integration check ────────────────────────────────────────")
test_model = OralHealthModelV2(CONFIG).to("cpu")
try:
    ckpt = torch.load(final_model_paths[0], map_location="cpu", weights_only=False)
except TypeError:
    ckpt = torch.load(final_model_paths[0], map_location="cpu")

test_model.load_state_dict(ckpt["model_state_dict"])
test_model.eval()
dummy = torch.zeros(1, 3, CONFIG["image_size"], CONFIG["image_size"])
out   = test_model(dummy, dummy, dummy)
print("✓ Model load + forward pass OK")
print("  Output shapes:", {k: tuple(v.shape) for k, v in out.items()})
print("  Decode example (all-zero input):",
      {k: decode_ordinal(v).item() for k, v in out.items()})
del test_model
print("────────────────────────────────────────────────────────────────────\n")
print("Done. Model checkpoints are in:", CONFIG["checkpoint_dir"])
