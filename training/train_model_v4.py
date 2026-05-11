#!/usr/bin/env python3
"""
DentAI Oral Health Prediction — Training Script v4
====================================================
Maximally accurate training for MGI (0-4), OHI (0-2 merged), GEI (0-2 merged).

ARCHITECTURE:
  • DINOv2-small: CLS token + patch-avg concatenated (768d) with cross-view attention
  • EfficientNet-B4: cross-view attention (1792d)
  • Fused: 2560d → projected to 256d
  • Per-task ordinal heads (K-1 binary thresholds each)
  • Per-task SupCon projection heads (3 × independent)
  • GEI binary auxiliary head ("any enlargement?")

LOSS STACK:
  • OrdinalBCE       — pos_weight FLOOR=1.0 (prevents cheap-positive collapse)
  • AsymmetricLoss   — γ_neg=3.0 >> γ_pos=0.5 (down-weights easy negatives)
  • OrdinalMAELoss   — |E[pred]-true| (blocks mode-collapse to class 1)
  • MonotonicityLoss — P(Y≥k) ≥ P(Y≥k+1) penalty (ordinal consistency)
  • SupConLoss       — per-task feature-space class separation
  • GEIBinaryAux     — dedicated 0/1 boundary classifier
  • RDropLoss        — dual-forward KL consistency

TRAINING STRATEGY:
  • 5-fold stratified CV, 150 epochs each
  • Freeze backbone 10 epochs → unfreeze last 6 blocks
  • EMA evaluation from epoch 12 onwards
  • SWA averaging from epoch 110
  • MixUp (epoch-based ramp) + CutMix interleaved
  • Hard-example replay buffer (rare-class patients replayed 30% extra)
  • Adaptive augmentation strength (rare classes get 1.4× intensity)
  • SGDR (CosineWarmRestarts) scheduler after unfreeze
  • Per-threshold calibration (K-1 independent thresholds, not single scalar)
  • Temperature scaling (post-hoc probability calibration via LBFGS on val)
  • All-fold SWA ensemble with TTA at inference

PLACE:   <project_root>/training/train_model_v4.py
RUN:     python training/train_model_v4.py
VRAM:    ≥8 GB  (batch=4, grad_accum=4 → effective_batch=16)
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Imports
# ─────────────────────────────────────────────────────────────────────────────
import os, re, json, random, math, logging, warnings, time, shutil
import multiprocessing as mp
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
from sklearn.metrics import (f1_score, confusion_matrix,
                             classification_report, accuracy_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


seed_everything(42)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Configuration
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    # ── Image ─────────────────────────────────────────────────────────
    "image_size":               336,

    # ── Classes (OHI/GEI merged: 2+3 → 2) ────────────────────────────
    "num_classes_mgi":           5,   # 0-4 unchanged
    "num_classes_ohi":           3,   # 0, 1, 2+ merged
    "num_classes_gei":           3,   # 0, 1, 2+ merged

    # ── Backbones ─────────────────────────────────────────────────────
    "backbone_dino":    "vit_small_patch14_dinov2.lvd142m",
    "backbone_cnn":     "efficientnet_b4",
    "pretrained":                True,
    "freeze_epochs":             10,
    "unfreeze_blocks":            6,

    # ── Architecture ──────────────────────────────────────────────────
    "dropout":                   0.25,
    "projection_dim":            256,
    "head_hidden_dim":           128,
    "view_attn_heads":             4,
    "dino_use_cls":              True,   # CLS+patch_avg concat → 768d
    "cnn_cross_view_attn":       True,   # cross-attn on CNN features too
    "supcon_dim":                128,    # per-task SupCon embedding dim
    "supcon_temp":               0.07,

    # ── Training ──────────────────────────────────────────────────────
    "batch_size":                  4,
    "grad_accum_steps":            4,
    "num_epochs":                150,
    "warmup_epochs":               5,

    # ── Learning rates ────────────────────────────────────────────────
    "lr_backbone":             3e-7,
    "lr_projection":           3e-5,
    "lr_heads":                1e-4,
    "weight_decay":            1e-4,
    "grad_clip_norm":           1.0,

    # ── Scheduler — SGDR with 2 warm restarts ─────────────────────────
    "sgdr_T0":                   30,    # first cycle length (epochs)
    "sgdr_T_mult":                2,    # doubles after each restart
    "sgdr_eta_min_frac":         0.05,  # eta_min = lr * this

    # ── EMA + SWA ─────────────────────────────────────────────────────
    "ema_decay":               0.998,
    "ema_start_epoch":            12,
    "use_ema_eval":              True,
    "swa_start_epoch":           110,

    # ── Loss weights ──────────────────────────────────────────────────
    "pw_floor":                  1.0,   # CRITICAL: pos_weight ≥ 1 always
    "pw_ceil":                   8.0,
    "alpha_ordinal":             1.2,
    "alpha_asl":                 0.6,
    "alpha_mae":                 0.4,
    "alpha_mono":                0.3,   # monotonicity penalty
    "alpha_supcon":              0.4,
    "alpha_gei_aux":             0.2,
    "alpha_rdrop":               0.2,
    "asl_gamma_neg":             3.0,
    "asl_gamma_pos":             0.5,
    "asl_clip":                  0.05,
    "task_loss_weights":         {"mgi": 0.45, "ohi": 0.30, "gei": 0.25},

    # ── Sampler ───────────────────────────────────────────────────────
    "hard_replay_frac":          0.30,
    "hard_replay_start_epoch":   12,

    # ── MixUp + CutMix ────────────────────────────────────────────────
    "mixup_alpha":               0.4,
    "mixup_prob_start":          0.20,
    "mixup_prob_end":            0.45,
    "cutmix_alpha":              1.0,
    "cutmix_prob":               0.30,

    # ── Per-threshold calibration ─────────────────────────────────────
    "calib_grid_steps":          60,
    "calib_every_epochs":        10,
    "calib_mgi_range":         (0.20, 0.90),
    "calib_ohi_range":         (0.20, 0.85),
    "calib_gei_range":         (0.15, 0.90),

    # ── Temperature scaling ───────────────────────────────────────────
    "temp_scale":                True,
    "temp_lr":                   0.01,
    "temp_max_iter":             500,

    # ── Checkpoint objective ──────────────────────────────────────────
    "ckpt_mae_penalty":          0.10,  # obj = avg_F1 - penalty*norm_MAE

    # ── TTA ───────────────────────────────────────────────────────────
    "tta_steps":                   7,

    # ── K-fold ────────────────────────────────────────────────────────
    "k_folds":                     5,
    "early_stopping_patience":    30,
    "seed":                       42,

    # ── DataLoader ────────────────────────────────────────────────────
    "num_workers":                 0,   # 0 = Windows-safe

    # ── Paths ─────────────────────────────────────────────────────────
    "data_dir":       str(PROJECT_ROOT / "Thesis_Data"),
    "checkpoint_dir": str(PROJECT_ROOT / "models" / "checkpoints"),
    "plots_dir":      str(PROJECT_ROOT / "outputs"  / "plots"),
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Label merging
# ─────────────────────────────────────────────────────────────────────────────

def merge_ohi(x: int) -> int: return min(int(x), 2)
def merge_gei(x: int) -> int: return min(int(x), 2)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CSV parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_thesis_csv(data_dir: str):
    root     = Path(data_dir)
    csv_path = root / "Thesis_Results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=[0, 1, 2], dtype=str)
    photo_root = root / "Thesis_Photographs"
    view_map = {
        "frontal": (photo_root / "Frontal",      "F"),
        "left":    (photo_root / "Left_Lateral", "L"),
        "right":   (photo_root / "Right_Lateral","R"),
    }

    records, patient_labels, skipped = [], {}, 0
    for _, row in df.iterrows():
        try:
            pid  = int(str(row.iloc[0]).strip())
            stxt = str(row.iloc[2]).strip()
        except Exception:
            skipped += 1; continue

        mm = re.search(r"MGI\s*-\s*(\d+)", stxt, re.I)
        om = re.search(r"OHI\s*-\s*(\d+)", stxt, re.I)
        gm = re.search(r"GEI\s*-\s*(\d+)", stxt, re.I)
        if not (mm and om and gm):
            skipped += 1; continue

        mgi = int(mm.group(1))
        ohi = merge_ohi(int(om.group(1)))
        gei = merge_gei(int(gm.group(1)))
        if not (0 <= mgi <= 4 and 0 <= ohi <= 2 and 0 <= gei <= 2):
            skipped += 1; continue

        patient_labels[str(pid)] = {"mgi": mgi, "ohi": ohi, "gei": gei}
        for vname, (folder, pfx) in view_map.items():
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                c = folder / f"{pfx}{pid}{ext}"
                if c.exists():
                    img_path = c; break
            if img_path is None:
                continue
            records.append({
                "patient_id": str(pid), "view": vname,
                "image_path": str(img_path),
                "mgi": mgi, "ohi": ohi, "gei": gei,
            })

    print(f"CSV: {len(patient_labels)} patients  "
          f"{len(records)} images  ({skipped} skipped)")
    return records, patient_labels


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Pos-weight computation  (floor = 1.0, critical)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pos_weights(patient_labels: dict, config: dict) -> dict:
    """
    For threshold k of task T:
        pos_weight = clamp( neg_count / pos_count, floor=1.0, ceil=8.0 )

    floor=1.0 ensures positive predictions are never cheaper than negative.
    In v2 this was 0.57 for MGI-threshold-1, causing 100% class-1 predictions.
    """
    n     = len(patient_labels)
    floor = config["pw_floor"]
    ceil_ = config["pw_ceil"]

    task_cfg = [
        ("mgi", config["num_classes_mgi"]),
        ("ohi", config["num_classes_ohi"]),
        ("gei", config["num_classes_gei"]),
    ]
    lbls = {t: [v[t] for v in patient_labels.values()] for t, _ in task_cfg}

    pw_dict = {}
    print("── POS-WEIGHTS (floor=1.0 enforced) ────────────────────────────")
    for task, nc in task_cfg:
        pw = []
        for k in range(1, nc):
            pos = sum(1 for y in lbls[task] if y >= k)
            neg = n - pos
            w   = float(np.clip(neg / max(pos, 1), floor, ceil_))
            pw.append(w)
        pw_dict[task] = torch.tensor(pw, dtype=torch.float32)
        print(f"  {task}: {['%.2f'%w for w in pw]}")
    print("─────────────────────────────────────────────────────────────────\n")
    return pw_dict


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Augmentation
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
_SZ = CONFIG["image_size"]


def _build_aug(strength: float = 1.0) -> A.Compose:
    s = strength
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.05),
        A.ShiftScaleRotate(shift_limit=0.06 * s, scale_limit=0.18 * s,
                           rotate_limit=int(28 * s), p=0.70),
        A.RandomResizedCrop(height=_SZ, width=_SZ,
                            scale=(max(0.65, 0.80 - 0.15 * s), 1.0),
                            ratio=(0.85, 1.15), p=0.65),
        A.GridDistortion(num_steps=3, distort_limit=0.25 * s, p=0.20),
        A.ElasticTransform(alpha=int(60 * s), sigma=6, p=0.15),
        # Colour — gingival redness is the primary clinical discriminator
        A.RandomBrightnessContrast(
            brightness_limit=min(0.40 * s, 0.50),
            contrast_limit  =min(0.40 * s, 0.50), p=0.70),
        A.HueSaturationValue(hue_shift_limit=int(18 * s),
                             sat_shift_limit=int(45 * s),
                             val_shift_limit=int(35 * s), p=0.65),
        A.RGBShift(r_shift_limit=int(25 * s),
                   g_shift_limit=int(18 * s),
                   b_shift_limit=int(18 * s), p=0.45),
        A.ColorJitter(brightness=0.25 * s, contrast=0.25 * s,
                      saturation=0.30 * s, hue=0.08 * s, p=0.50),
        A.CLAHE(clip_limit=3.5 * s, tile_grid_size=(8, 8), p=0.40),
        A.RandomShadow(p=0.10),
        A.GaussianBlur(blur_limit=(3, max(3, int(7 * s) // 2 * 2 + 1)), p=0.25),
        A.GaussNoise(var_limit=(10, int(60 * s)), p=0.30),
        A.ImageCompression(quality_lower=max(50, int(70 - 20 * s)),
                           quality_upper=100, p=0.20),
        A.CoarseDropout(max_holes=6, max_height=int(32 * s),
                        max_width =int(32 * s), min_holes=1, p=0.25),
        A.Resize(_SZ, _SZ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


TRAIN_AUG_NORMAL = _build_aug(1.0)
TRAIN_AUG_STRONG = _build_aug(1.4)   # rare-class patients get stronger aug
VAL_AUG = A.Compose([
    A.Resize(_SZ, _SZ),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _is_rare(s: dict) -> bool:
    """True if any label is in a minority class → stronger augmentation."""
    return (s["labels"]["mgi"] >= 3 or
            s["labels"]["ohi"] >= 2 or
            s["labels"]["gei"] >= 1)


class MultiViewPatientDataset(Dataset):
    """One sample = (frontal, left, right, {mgi, ohi, gei}).
    Only patients with all 3 views included.
    """

    REQUIRED = ("frontal", "left", "right")

    def __init__(self, records: list, transform=None, strong_transform=None):
        self.transform        = transform
        self.strong_transform = strong_transform

        grouped = defaultdict(lambda: {"views": {}})
        for r in records:
            pid = str(r["patient_id"])
            grouped[pid]["patient_id"] = pid
            grouped[pid]["labels"]     = {
                "mgi": r["mgi"], "ohi": r["ohi"], "gei": r["gei"]}
            grouped[pid]["views"][r["view"].lower()] = r["image_path"]

        self.samples = [
            s for s in grouped.values()
            if all(v in s["views"] for v in self.REQUIRED)
        ]
        self.samples.sort(
            key=lambda x: int(x["patient_id"]) if x["patient_id"].isdigit() else 0)

        self.mgi_labels = [s["labels"]["mgi"] for s in self.samples]
        self.ohi_labels = [s["labels"]["ohi"] for s in self.samples]
        self.gei_labels = [s["labels"]["gei"] for s in self.samples]
        print(f"Dataset: {len(self.samples)} complete 3-view patient triplets")

    def __len__(self): return len(self.samples)

    def _load(self, path: str) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((_SZ, _SZ, 3), dtype=np.uint8)
        img = cv2.resize(img, (_SZ, _SZ), interpolation=cv2.INTER_LANCZOS4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Gray-world white balance (compensates for clinic lighting variation)
        avg   = img.mean(axis=(0, 1)).astype(np.float32)
        scale = np.where(avg > 1e-6, avg.mean() / avg, 1.0)
        img   = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        # CLAHE on L-channel (improves gingival contrast)
        lab   = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge((cl.apply(l), a, b)), cv2.COLOR_LAB2RGB)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        aug = (self.strong_transform
               if (_is_rare(s) and self.strong_transform is not None)
               else self.transform)
        tensors = []
        for v in self.REQUIRED:
            img = self._load(s["views"][v])
            if aug is not None:
                img = aug(image=img)["image"]
            else:
                img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            tensors.append(img)
        return tensors[0], tensors[1], tensors[2], s["labels"]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class OrdinalBCELoss(nn.Module):
    """Binary CE over K-1 ordinal thresholds with pos_weight floor = 1.0."""

    def __init__(self, num_classes: int, pos_weight: torch.Tensor,
                 label_smooth: float = 0.02):
        super().__init__()
        self.K      = num_classes
        self.n_th   = num_classes - 1
        self.smooth = label_smooth
        self.register_buffer("pw", pos_weight)

    def _make_targets(self, y: torch.Tensor) -> torch.Tensor:
        B = y.shape[0]
        t = torch.zeros(B, self.n_th, device=y.device)
        for k in range(1, self.K):
            t[:, k - 1] = (y >= k).float()
        return t * (1.0 - self.smooth) + self.smooth * 0.5

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                lam: float = 1.0, targets_b=None) -> torch.Tensor:
        pw = self.pw.to(logits.device)
        t  = self._make_targets(targets)
        if targets_b is not None and lam < 1.0:
            t = lam * t + (1.0 - lam) * self._make_targets(targets_b)
        return F.binary_cross_entropy_with_logits(
            logits, t, pos_weight=pw, reduction="mean")


class AsymmetricLoss(nn.Module):
    """
    ASL (Hill et al. 2021): γ_neg >> γ_pos.
    Aggressively down-weights easy healthy-class negatives so the model
    cannot coast on predicting healthy for everything.
    """

    def __init__(self, gamma_neg: float = 3.0, gamma_pos: float = 0.5,
                 clip: float = 0.05):
        super().__init__()
        self.gn   = gamma_neg
        self.gp   = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        p     = torch.sigmoid(logits)
        p_neg = (p - self.clip).clamp(min=0)
        lp    = (1.0 - p) ** self.gp * F.logsigmoid( logits)
        ln    = p_neg     ** self.gn * F.logsigmoid(-logits)
        return -(targets * lp + (1.0 - targets) * ln).mean()


class OrdinalMAELoss(nn.Module):
    """
    Huber-smooth L1 on |E[predicted_class] - true_class|.
    Blocks mode-collapse: if model always predicts class 1, E[pred]=1 and
    every true class ≠ 1 fires a gradient pushing the model away from class 1.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                lam: float = 1.0, targets_b=None) -> torch.Tensor:
        expected = torch.sigmoid(logits).sum(dim=1)   # E[ordinal class]
        y = targets.float()
        if targets_b is not None and lam < 1.0:
            y = lam * y + (1.0 - lam) * targets_b.float()
        return F.smooth_l1_loss(expected, y, beta=0.5)


class MonotonicityLoss(nn.Module):
    """
    Penalises P(Y >= k+1) > P(Y >= k).
    Ordinal BCE optimises each threshold independently; without this loss
    non-monotonic probabilities arise and break the ordinal decoding.
    """

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] < 2:
            return logits.new_zeros(())
        p    = torch.sigmoid(logits)                   # (B, K-1)
        viol = F.relu(p[:, 1:] - p[:, :-1])           # violations
        return viol.pow(2).mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. 2020).
    Adjacent ordinal classes treated as soft positives (weight adj_weight).
    """

    def __init__(self, temperature: float = 0.07, adj_weight: float = 0.3):
        super().__init__()
        self.temp = temperature
        self.adj  = adj_weight

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        if B < 2:
            return features.new_zeros(())

        sim      = torch.mm(features, features.T) / self.temp  # (B, B)
        lr       = labels.unsqueeze(1).expand(B, B)
        lc       = labels.unsqueeze(0).expand(B, B)
        same     = (lr == lc).float()
        adj      = ((lr - lc).abs() == 1).float() * self.adj
        mask     = (same + adj).clamp(max=1.0)
        diag     = torch.eye(B, device=features.device)
        mask     = mask * (1.0 - diag)
        vpos     = mask.sum(1).clamp(min=1e-8)
        exp_sim  = torch.exp(sim) * (1.0 - diag)
        log_den  = torch.log(exp_sim.sum(1).clamp(min=1e-8))
        log_prob = sim - log_den.unsqueeze(1)
        return (-(mask * log_prob).sum(1) / vpos).mean()


class GEIBinaryAux(nn.Module):
    """Binary 'any gingival enlargement?' head for the GEI 0/1 boundary."""

    def __init__(self, pos_weight: float = 5.5):
        super().__init__()
        self.pw = pos_weight

    def forward(self, logit: torch.Tensor,
                gei_labels: torch.Tensor) -> torch.Tensor:
        target = (gei_labels >= 1).float()
        pw     = logit.new_tensor([self.pw])
        return F.binary_cross_entropy_with_logits(
            logit.squeeze(1), target, pos_weight=pw)


class RDropLoss(nn.Module):
    """
    R-Drop (Liang et al. 2021): two forward passes with different dropout masks.
    KL divergence between the two output distributions forces prediction
    consistency — strong regulariser for small datasets (203 patients).
    """

    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    @staticmethod
    def _to_probs(logits: torch.Tensor) -> torch.Tensor:
        """(B, K-1) ordinal logits → (B, K) class probability distribution."""
        cum   = torch.sigmoid(logits)
        ones  = torch.ones(*cum.shape[:-1], 1, device=logits.device)
        zeros = torch.zeros(*cum.shape[:-1], 1, device=logits.device)
        full  = torch.cat([ones, cum, zeros], dim=-1)
        p     = (full[..., :-1] - full[..., 1:]).clamp(min=1e-8)
        return p / p.sum(-1, keepdim=True)

    def forward(self, preds1: dict, preds2: dict) -> torch.Tensor:
        total = next(iter(preds1.values())).new_zeros(())
        for task in preds1:
            p1 = self._to_probs(preds1[task])
            p2 = self._to_probs(preds2[task])
            kl = (F.kl_div(p1.log(), p2, reduction="batchmean") +
                  F.kl_div(p2.log(), p1, reduction="batchmean")) * 0.5
            total = total + kl
        return self.alpha * total / len(preds1)


class MultiTaskLossV4(nn.Module):
    """
    Full multi-task loss combining all components for maximum accuracy.
    Each component targets a specific failure mode observed in earlier versions.
    """

    def __init__(self, pos_weights: dict, config: dict):
        super().__init__()
        sm = 0.02
        for task, nc in [("mgi", config["num_classes_mgi"]),
                         ("ohi", config["num_classes_ohi"]),
                         ("gei", config["num_classes_gei"])]:
            setattr(self, f"ord_{task}",  OrdinalBCELoss(nc, pos_weights[task], sm))
            setattr(self, f"mae_{task}",  OrdinalMAELoss())
            setattr(self, f"mono_{task}", MonotonicityLoss())

        self.asl     = AsymmetricLoss(config["asl_gamma_neg"],
                                       config["asl_gamma_pos"],
                                       config["asl_clip"])
        self.supcon  = SupConLoss(config["supcon_temp"])
        self.gei_aux = GEIBinaryAux(pos_weight=5.5)
        self.rdrop   = RDropLoss(config["alpha_rdrop"])

        self.a_ord   = config["alpha_ordinal"]
        self.a_asl   = config["alpha_asl"]
        self.a_mae   = config["alpha_mae"]
        self.a_mono  = config["alpha_mono"]
        self.a_sup   = config["alpha_supcon"]
        self.a_geia  = config["alpha_gei_aux"]
        self.tw      = config["task_loss_weights"]

    @staticmethod
    def _bce_targets(targets: torch.Tensor, K: int, lam: float = 1.0,
                     tb=None, s: float = 0.02) -> torch.Tensor:
        B = targets.shape[0]
        t = torch.zeros(B, K - 1, device=targets.device)
        for k in range(1, K):
            t[:, k - 1] = (targets >= k).float()
        t = t * (1.0 - s) + s * 0.5
        if tb is not None and lam < 1.0:
            t2 = torch.zeros_like(t)
            for k in range(1, K):
                t2[:, k - 1] = (tb >= k).float()
            t2 = t2 * (1.0 - s) + s * 0.5
            t  = lam * t + (1.0 - lam) * t2
        return t

    def forward(self,
                preds:           dict,
                targets:         dict,
                nc_map:          dict,
                feat_per_task:   dict  = None,
                gei_aux_logit:   torch.Tensor = None,
                preds2:          dict  = None,
                lam:             float = 1.0,
                targets_b:       dict  = None):
        info  = {}
        total = next(iter(preds.values())).new_zeros(())

        for task, nc in nc_map.items():
            logits = preds[task]
            tgt    = targets[task]
            tb     = targets_b[task] if targets_b else None

            l_ord  = getattr(self, f"ord_{task}")(logits, tgt, lam, tb)
            l_mae  = getattr(self, f"mae_{task}")(logits, tgt, lam, tb)
            l_mono = getattr(self, f"mono_{task}")(logits)

            bt     = self._bce_targets(tgt, nc, lam, tb)
            l_asl  = self.asl(logits, bt)

            l_t    = (self.a_ord * l_ord + self.a_asl * l_asl +
                      self.a_mae * l_mae + self.a_mono * l_mono)
            total  = total + self.tw[task] * l_t

            info[f"ord_{task}"]  = l_ord.item()
            info[f"mae_{task}"]  = l_mae.item()
            info[f"mono_{task}"] = l_mono.item()

        # SupCon (per-task, only on non-mixed batches)
        if feat_per_task is not None and lam >= 0.99:
            sc_list = []
            for task in nc_map:
                feat = feat_per_task.get(task)
                if feat is not None:
                    sc_list.append(self.supcon(feat, targets[task]))
            if sc_list:
                l_sc  = sum(sc_list) / len(sc_list)
                total = total + self.a_sup * l_sc
                info["supcon"] = l_sc.item()

        # GEI binary auxiliary
        if gei_aux_logit is not None:
            l_ga  = self.gei_aux(gei_aux_logit, targets["gei"])
            total = total + self.a_geia * l_ga
            info["gei_aux"] = l_ga.item()

        # R-Drop
        if preds2 is not None:
            l_rd  = self.rdrop(preds, preds2)
            total = total + l_rd
            info["rdrop"] = l_rd.item()

        return total, info


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Model
# ─────────────────────────────────────────────────────────────────────────────

def _build_dino(name: str, pretrained: bool, img_size: int) -> nn.Module:
    """
    Build DINOv2 backbone with global_pool='' so we get full token output.
    Handles the norm.weight → fc_norm.weight key mismatch in some timm builds.
    """
    def _mk(pre: bool):
        return timm.create_model(name, pretrained=pre,
                                 img_size=img_size,
                                 num_classes=0,
                                 global_pool="")    # returns (B, 1+N, D)
    try:
        return _mk(pretrained)
    except RuntimeError as exc:
        msg = str(exc)
        if "fc_norm.weight" in msg and "norm.weight" in msg:
            log.warning("DINOv2 key mismatch — remapping norm→fc_norm")
            m   = _mk(False)
            cfg = getattr(m, "pretrained_cfg", {}) or {}
            url = cfg.get("url")
            if not url:
                log.warning("No pretrained URL — using random init")
                return m
            sd    = torch.hub.load_state_dict_from_url(
                url, map_location="cpu", check_hash=False, progress=True)
            remap = {}
            for k, v in sd.items():
                if k == "norm.weight":   remap["fc_norm.weight"] = v
                elif k == "norm.bias":   remap["fc_norm.bias"]   = v
                elif k == "mask_token":  pass
                elif k == "pos_embed":
                    pe = getattr(m, "pos_embed", None)
                    if pe is not None and pe.shape != v.shape:
                        cls_ = v[:, :1, :]; ptok = v[:, 1:, :]
                        N0 = ptok.shape[1]; N1 = pe.shape[1] - 1
                        g0 = int(math.sqrt(N0)); g1 = int(math.sqrt(N1))
                        D  = ptok.shape[-1]
                        ptok = ptok.reshape(1, g0, g0, D).permute(0, 3, 1, 2)
                        ptok = F.interpolate(ptok, (g1, g1), mode="bicubic",
                                             align_corners=False)
                        ptok = ptok.permute(0, 2, 3, 1).reshape(1, N1, D)
                        remap[k] = torch.cat([cls_, ptok], dim=1)
                    else:
                        remap[k] = v
                else:
                    remap[k] = v
            incomp = m.load_state_dict(remap, strict=False)
            if incomp.missing_keys:
                log.info(f"Remap missing: {incomp.missing_keys[:4]}")
            return m
        if pretrained:
            log.warning(f"DINOv2 load failed: {exc}. Using random init.")
            return _mk(False)
        raise


def _extract_dino(backbone: nn.Module, x: torch.Tensor,
                  use_cls: bool) -> torch.Tensor:
    """
    Run DINOv2, return:
      use_cls=True  → cat(CLS, patch_avg) → (B, 2D)
      use_cls=False → patch_avg           → (B, D)
    """
    out = backbone(x)           # (B, 1+N_patches, D) or (B, D) fallback
    if out.dim() == 2:
        return out              # fallback: backbone already pooled
    cls_tok    = out[:, 0, :]
    patch_avg  = out[:, 1:, :].mean(1)
    if use_cls:
        return torch.cat([cls_tok, patch_avg], dim=1)
    return patch_avg


class ViewAttentionPool(nn.Module):
    """
    Self-attention over 3 intra-oral view tokens → mean-pool to single vector.
    Lets frontal, left, right views inform each other before prediction.
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.10):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, heads,
                                            batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, D)
        a, _ = self.attn(x, x, x)
        x    = self.norm1(x + a)
        return self.norm2(x + self.ff(x)).mean(1)  # (B, D)


class _TaskHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, drop=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(drop), nn.Linear(hidden, out_dim))

    def forward(self, x): return self.net(x)


class OralHealthModelV4(nn.Module):
    """
    DINOv2-small (CLS+patch=768d) + EfficientNet-B4 (1792d)
    Both with cross-view attention → concat → project → per-task heads
    Fused dim: 768 + 1792 = 2560 → 256 projected
    """

    def __init__(self, config: dict):
        super().__init__()
        self.use_cls  = config.get("dino_use_cls", True)
        self.cnn_attn = config.get("cnn_cross_view_attn", True)

        # ── Backbones ──────────────────────────────────────────────────
        self.dino = _build_dino(config["backbone_dino"],
                                 config["pretrained"],
                                 config["image_size"])
        self.cnn  = timm.create_model(config["backbone_cnn"],
                                       pretrained=config["pretrained"],
                                       num_classes=0, global_pool="avg")

        dino_base = self.dino.num_features         # 384 for DINOv2-small
        dino_dim  = dino_base * 2 if self.use_cls else dino_base
        cnn_dim   = self.cnn.num_features          # 1792 for EfficientNet-B4
        fused_dim = dino_dim + cnn_dim             # 2560

        # ── Cross-view attention pools ─────────────────────────────────
        self.dino_pool = ViewAttentionPool(dino_dim, config["view_attn_heads"])
        if self.cnn_attn:
            self.cnn_pool = ViewAttentionPool(cnn_dim, config["view_attn_heads"])

        # ── Shared projection ──────────────────────────────────────────
        proj_dim = config["projection_dim"]
        self.proj = nn.Sequential(
            nn.Linear(fused_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
        )

        # ── Per-task SupCon projection heads ───────────────────────────
        sc_dim = config["supcon_dim"]
        for t in ("mgi", "ohi", "gei"):
            setattr(self, f"sc_proj_{t}",
                    nn.Sequential(nn.Linear(proj_dim, sc_dim),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(sc_dim, sc_dim)))

        # ── Ordinal task heads ─────────────────────────────────────────
        hd = config["head_hidden_dim"]
        dp = config["dropout"]
        self.mgi_head = _TaskHead(proj_dim, config["num_classes_mgi"] - 1, hd, dp)
        self.ohi_head = _TaskHead(proj_dim, config["num_classes_ohi"] - 1, hd, dp)
        self.gei_head = _TaskHead(proj_dim, config["num_classes_gei"] - 1, hd, dp)

        # ── GEI binary auxiliary ───────────────────────────────────────
        self.gei_aux  = _TaskHead(proj_dim, 1, 64, dp)

    # ------------------------------------------------------------------
    def _dino_feat(self, x: torch.Tensor) -> torch.Tensor:
        return _extract_dino(self.dino, x, self.use_cls)

    def _cnn_feat(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)

    def _fuse_views(self, f, l, r) -> torch.Tensor:
        # DINOv2 cross-view attention
        d_stack = torch.stack([self._dino_feat(f),
                               self._dino_feat(l),
                               self._dino_feat(r)], dim=1)    # (B, 3, dino_dim)
        d_out   = self.dino_pool(d_stack)                     # (B, dino_dim)

        # CNN cross-view attention
        c_stack = torch.stack([self._cnn_feat(f),
                               self._cnn_feat(l),
                               self._cnn_feat(r)], dim=1)     # (B, 3, cnn_dim)
        c_out   = self.cnn_pool(c_stack) if self.cnn_attn else c_stack.mean(1)

        return torch.cat([d_out, c_out], dim=1)               # (B, fused_dim)

    def forward(self, frontal, left, right,
                return_features: bool = False):
        fused = self._fuse_views(frontal, left, right)
        proj  = self.proj(fused)

        out = {
            "mgi":     self.mgi_head(proj),
            "ohi":     self.ohi_head(proj),
            "gei":     self.gei_head(proj),
            "gei_aux": self.gei_aux(proj),
        }

        if return_features:
            # Per-task L2-normalised embeddings for SupCon
            out["features"] = {
                t: F.normalize(getattr(self, f"sc_proj_{t}")(proj), dim=1)
                for t in ("mgi", "ohi", "gei")
            }

        return out

    # ------------------------------------------------------------------
    def freeze_backbone(self):
        for p in (list(self.dino.parameters()) + list(self.cnn.parameters())):
            p.requires_grad_(False)
        log.info("Backbones frozen.")

    def unfreeze_last_n_blocks(self, n: int):
        # DINOv2 transformer blocks
        for blk in list(self.dino.blocks)[-n:]:
            for p in blk.parameters(): p.requires_grad_(True)
        for nm, mod in self.dino.named_modules():
            if "norm" in nm and "blocks" not in nm:
                for p in mod.parameters(): p.requires_grad_(True)
        # EfficientNet top-level children
        for ch in list(self.cnn.children())[-n:]:
            for p in ch.parameters(): p.requires_grad_(True)
        n_tr = sum(p.requires_grad for p in
                   list(self.dino.parameters()) + list(self.cnn.parameters()))
        log.info(f"Unfroze last {n} blocks → {n_tr} backbone params trainable.")

    def head_parameters(self):
        mods = [self.dino_pool, self.proj,
                self.mgi_head, self.ohi_head, self.gei_head, self.gei_aux]
        if self.cnn_attn: mods.append(self.cnn_pool)
        for t in ("mgi", "ohi", "gei"):
            mods.append(getattr(self, f"sc_proj_{t}"))
        return [p for m in mods for p in m.parameters()]

    def backbone_parameters(self):
        return [p for p in (list(self.dino.parameters()) + list(self.cnn.parameters()))
                if p.requires_grad]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Decoding
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_ordinal(logits: torch.Tensor,
                   thresholds=None,
                   temperature: float = 1.0) -> torch.Tensor:
    """
    Decode (B, K-1) ordinal logits → (B,) integer class predictions.

    thresholds: None → 0.5 scalar
                float → same scalar for all thresholds
                list[float] of length K-1 → per-threshold (v4 new)
    temperature: divide logits before sigmoid (for temp-scaling calibration)
    """
    probs = torch.sigmoid(logits / max(float(temperature), 1e-4))
    if thresholds is None or isinstance(thresholds, float):
        thr = float(thresholds) if thresholds is not None else 0.5
        return (probs > thr).sum(dim=1).long()
    # Per-threshold list
    thr = torch.tensor(list(thresholds), dtype=probs.dtype, device=probs.device)
    return (probs > thr.unsqueeze(0)).sum(dim=1).long()


@torch.no_grad()
def ensemble_tta_predict(models_list: list,
                          frontal, left, right,
                          thresholds_list: list,
                          temps_list: list,
                          n_tta: int = 7) -> dict:
    """
    Run TTA over a list of models and average the sigmoid probabilities.
    Used for the all-fold SWA ensemble at final evaluation.
    """
    accum = {"mgi": None, "ohi": None, "gei": None}

    def _aug(t):
        if random.random() > 0.5:
            t = torch.flip(t, dims=[-1])
        return (t + torch.randn_like(t) * 0.008).clamp(-3, 3)

    for model, thr_dict, temp in zip(models_list, thresholds_list, temps_list):
        model.eval()
        for tta_i in range(n_tta):
            fi = frontal if tta_i == 0 else _aug(frontal.clone())
            li = left    if tta_i == 0 else _aug(left.clone())
            ri = right   if tta_i == 0 else _aug(right.clone())

            out = model(fi, li, ri)
            for task in ("mgi", "ohi", "gei"):
                p = torch.sigmoid(out[task] / max(float(temp), 1e-4))
                if accum[task] is None:
                    accum[task] = p.clone()
                else:
                    accum[task] = accum[task] + p

    n_total = len(models_list) * n_tta
    results = {}
    for task in ("mgi", "ohi", "gei"):
        avg_p = accum[task] / n_total
        # Use averaged thresholds across all fold models
        avg_thr = thresholds_list[0].get(task, 0.5)  # already averaged externally
        results[task] = decode_ordinal(
            torch.log(avg_p.clamp(1e-7) / (1.0 - avg_p).clamp(1e-7)),
            avg_thr)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — Samplers + data helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_sampler(dataset: MultiViewPatientDataset,
                  indices: list) -> WeightedRandomSampler:
    """
    Multi-task geometric-mean sampler with 2× boost for hardest minorities.
    Ensures all three tasks see balanced batches simultaneously.
    """
    get = lambda t: np.array([getattr(dataset, f"{t}_labels")[i] for i in indices])
    ma, oa, ga = get("mgi"), get("ohi"), get("gei")

    def inv_freq(arr, nc):
        c = np.bincount(arr, minlength=nc).astype(float)
        c = np.where(c == 0, 0.5, c)
        return np.array([1.0 / c[y] for y in arr])

    w = (inv_freq(ma, 5) * inv_freq(oa, 3) * inv_freq(ga, 3)) ** (1.0 / 3.0)
    # Extra boost for rarest clinical groups
    boost = np.where((ma >= 3) | (ga >= 1), 2.0, 1.0)
    w     = w * boost
    w     = np.clip(w, None, 6.0 * np.median(w))
    w     = w / w.sum()
    return WeightedRandomSampler(
        torch.tensor(w, dtype=torch.float32), len(indices), replacement=True)


def build_hard_replay(dataset: MultiViewPatientDataset, indices: list) -> list:
    return [i for i in indices
            if (dataset.mgi_labels[i] >= 3 or
                dataset.ohi_labels[i] >= 2 or
                dataset.gei_labels[i] >= 1)]


def make_epoch_indices(base: list, pool: list, frac: float) -> list:
    if frac <= 0 or not pool:
        return list(base)
    n   = max(1, int(round(len(base) * frac)))
    rep = np.random.choice(pool, n, replace=(len(pool) < n)).tolist()
    return list(base) + rep


def _collect_batch(batch, device):
    f, l, r, labels = batch
    targets = {t: labels[t].to(device).long() for t in ("mgi", "ohi", "gei")}
    return f.to(device), l.to(device), r.to(device), targets


def _mixup(f, l, r, targets, alpha: float, prob: float):
    if alpha <= 0 or random.random() > prob:
        return f, l, r, targets, None, 1.0
    lam = max(float(np.random.beta(alpha, alpha)), 0.5)
    B   = f.shape[0]
    idx = torch.randperm(B, device=f.device)
    tb  = {k: v[idx] for k, v in targets.items()}
    return (lam * f + (1 - lam) * f[idx],
            lam * l + (1 - lam) * l[idx],
            lam * r + (1 - lam) * r[idx],
            targets, tb, lam)


def _cutmix(f, l, r, targets, alpha: float, prob: float):
    """
    CutMix: paste a random rectangular crop from one patient onto another.
    Better than MixUp for dental photos since local gingival texture matters.
    """
    if alpha <= 0 or random.random() > prob:
        return f, l, r, targets, None, 1.0
    lam = float(np.random.beta(alpha, alpha))
    B, C, H, W = f.shape
    idx = torch.randperm(B, device=f.device)

    cut_h = max(1, int(H * math.sqrt(1.0 - lam)))
    cut_w = max(1, int(W * math.sqrt(1.0 - lam)))
    cx    = random.randint(0, W)
    cy    = random.randint(0, H)
    x1    = max(0, cx - cut_w // 2); x2 = min(W, cx + cut_w // 2)
    y1    = max(0, cy - cut_h // 2); y2 = min(H, cy + cut_h // 2)

    fm = f.clone(); lm = l.clone(); rm = r.clone()
    fm[:, :, y1:y2, x1:x2] = f[idx, :, y1:y2, x1:x2]
    lm[:, :, y1:y2, x1:x2] = l[idx, :, y1:y2, x1:x2]
    rm[:, :, y1:y2, x1:x2] = r[idx, :, y1:y2, x1:x2]

    # Recompute lambda from actual pasted area
    lam_real = 1.0 - (x2 - x1) * (y2 - y1) / float(H * W)
    tb = {k: v[idx] for k, v in targets.items()}
    return fm, lm, rm, targets, tb, lam_real


def _mixup_prob(epoch: int, config: dict) -> float:
    if epoch <= config["freeze_epochs"]:
        return 0.0
    p0    = config.get("mixup_prob_start", 0.20)
    p1    = config.get("mixup_prob_end",   0.45)
    start = config["freeze_epochs"] + 1
    t     = (epoch - start) / max(config["num_epochs"] - start, 1)
    return float(np.clip(p0 + (p1 - p0) * t, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — AMP utilities
# ─────────────────────────────────────────────────────────────────────────────

def _make_scaler():
    if hasattr(torch.amp, "GradScaler"):
        try:    return torch.amp.GradScaler("cuda")
        except: return torch.amp.GradScaler()
    return _amp_compat.GradScaler()


def _autocast():
    if hasattr(torch.amp, "autocast"):
        try:    return torch.amp.autocast(device_type="cuda", enabled=True)
        except: return torch.amp.autocast("cuda", enabled=True)
    return _amp_compat.autocast(enabled=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — Train / eval epoch functions
# ─────────────────────────────────────────────────────────────────────────────

def run_train_epoch(model, loader, optimizer, loss_fn, scaler,
                    device, config, phase, epoch, ema=None):
    model.train()
    totals  = {"total": 0.0, "mgi": 0.0, "ohi": 0.0, "gei": 0.0}
    n_steps = 0
    accum   = config["grad_accum_steps"]
    nc_map  = {"mgi": config["num_classes_mgi"],
               "ohi": config["num_classes_ohi"],
               "gei": config["num_classes_gei"]}

    mx_p  = _mixup_prob(epoch, config) if phase == 2 else 0.0
    cut_p = config["cutmix_prob"]      if phase == 2 else 0.0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        f, l, r, targets = _collect_batch(batch, device)

        # Alternate MixUp / CutMix per step
        if phase == 2 and step % 2 == 0:
            f, l, r, targets, tb, lam = _mixup(
                f, l, r, targets, config["mixup_alpha"], mx_p)
        elif phase == 2:
            f, l, r, targets, tb, lam = _cutmix(
                f, l, r, targets, config["cutmix_alpha"], cut_p)
        else:
            tb, lam = None, 1.0

        with _autocast():
            # Two forward passes: one with SupCon features, one for R-Drop
            ret_feat = (lam >= 0.99)   # SupCon only on non-mixed batches
            out1     = model(f, l, r, return_features=ret_feat)
            feat     = out1.get("features")    # dict {mgi,ohi,gei} or None
            pred1    = {k: out1[k] for k in ("mgi", "ohi", "gei")}
            g_aux    = out1["gei_aux"]

            out2     = model(f, l, r, return_features=False)
            pred2    = {k: out2[k] for k in ("mgi", "ohi", "gei")}

            total_loss, info = loss_fn(
                pred1, targets, nc_map,
                feat_per_task  = feat,
                gei_aux_logit  = g_aux,
                preds2         = pred2,
                lam            = lam,
                targets_b      = tb,
            )

        # ── Correct AMP backward (scaler.scale wraps the backward call) ──
        scaler.scale(total_loss / accum).backward()

        if (step + 1) % accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        totals["total"] += total_loss.item()
        for k in ("mgi", "ohi", "gei"):
            totals[k] += info.get(f"ord_{k}", 0.0)
        n_steps += 1

    # Flush any leftover gradient accumulation
    if n_steps % accum != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        optimizer.zero_grad(set_to_none=True)

    return {k: v / max(n_steps, 1) for k, v in totals.items()}


def run_val_epoch(model, loader, loss_fn, device, config,
                  thresholds: dict = None, temperature: float = 1.0):
    model.eval()
    y_true   = {"mgi": [], "ohi": [], "gei": []}
    y_pred   = {"mgi": [], "ohi": [], "gei": []}
    tot_loss = 0.0
    n        = 0
    nc_map   = {"mgi": config["num_classes_mgi"],
                "ohi": config["num_classes_ohi"],
                "gei": config["num_classes_gei"]}

    if thresholds is None:
        thresholds = {t: [0.5] * (nc_map[t] - 1) for t in nc_map}

    with torch.no_grad():
        for batch in loader:
            f, l, r, targets = _collect_batch(batch, device)
            with _autocast():
                out  = model(f, l, r)
                pred = {k: out[k] for k in ("mgi", "ohi", "gei")}
                loss, _ = loss_fn(pred, targets, nc_map,
                                   gei_aux_logit=out["gei_aux"])
            tot_loss += loss.item()
            n        += 1

            for task in ("mgi", "ohi", "gei"):
                y_true[task].extend(targets[task].cpu().numpy().tolist())
                y_pred[task].extend(
                    decode_ordinal(pred[task],
                                   thresholds.get(task, 0.5),
                                   temperature).cpu().numpy().tolist())

    return tot_loss / max(n, 1), y_true, y_pred


def compute_metrics(y_true: dict, y_pred: dict, nc_map: dict) -> dict:
    res = {}
    for task, nc in nc_map.items():
        yt = np.array(y_true[task])
        yp = np.array(y_pred[task])
        res[task] = {
            "f1":  f1_score(yt, yp, labels=list(range(nc)),
                             average="macro", zero_division=0),
            "acc": accuracy_score(yt, yp),
            "mae": float(np.mean(np.abs(yt.astype(int) - yp.astype(int)))),
        }
    return res


def ckpt_objective(metrics: dict, nc_map: dict, penalty: float = 0.10) -> float:
    """
    Composite checkpoint score = mean_macro_F1 − penalty × mean_norm_MAE.
    Normalising MAE by (K-1) puts it in [0, 1] regardless of task class count.
    Penalising large ordinal errors prevents the model from chasing F1 by
    collapsing distant classes.
    """
    f1s  = [metrics[t]["f1"]  for t in nc_map]
    maes = [metrics[t]["mae"] / max(nc_map[t] - 1, 1) for t in nc_map]
    return float(np.mean(f1s)) - penalty * float(np.mean(maes))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — Per-threshold calibration  (v4: each of K-1 boundaries tuned)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_per_threshold(model, val_loader, device, config,
                             temperature: float = 1.0) -> dict:
    """
    Grid-search each of the K-1 binary thresholds independently per task.

    Why per-threshold instead of a single scalar?
    The K-1 thresholds encode boundaries at different severity levels.
    MGI boundary-1 (healthy vs mild) has different optimal decision point
    than MGI boundary-4 (severe vs very-severe). A single threshold cannot
    optimise all of them simultaneously — this function finds each one.

    Returns: {task: [thr_k1, thr_k2, ..., thr_kKminus1]}
    """
    model.eval()
    all_probs  = {"mgi": [], "ohi": [], "gei": []}
    all_labels = {"mgi": [], "ohi": [], "gei": []}

    for batch in val_loader:
        f, l, r, targets = _collect_batch(batch, device)
        out = model(f, l, r)
        for task in ("mgi", "ohi", "gei"):
            p = torch.sigmoid(out[task] / max(float(temperature), 1e-4)).cpu()
            all_probs[task].append(p)
            all_labels[task].extend(targets[task].cpu().numpy().tolist())

    nc_map  = {"mgi": config["num_classes_mgi"],
               "ohi": config["num_classes_ohi"],
               "gei": config["num_classes_gei"]}
    t_range = {"mgi": config["calib_mgi_range"],
               "ohi": config["calib_ohi_range"],
               "gei": config["calib_gei_range"]}

    best_thrs = {}
    for task, (lo, hi) in t_range.items():
        probs  = torch.cat(all_probs[task], dim=0)   # (N, K-1)
        labels = np.array(all_labels[task])
        nc     = nc_map[task]
        Km1    = nc - 1
        grid   = np.linspace(lo, hi, config["calib_grid_steps"])
        thrs   = []

        # Start with all thresholds at 0.5, then optimise one-at-a-time
        current = [0.5] * Km1
        for k in range(Km1):
            best_f1 = -1.0
            best_t  = 0.5
            for thr in grid:
                current[k] = thr
                thr_vec    = torch.tensor(current, dtype=probs.dtype)
                preds      = (probs > thr_vec.unsqueeze(0)).sum(1).numpy()
                f1         = f1_score(labels, preds, labels=list(range(nc)),
                                      average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(thr)
            current[k] = best_t
            thrs.append(best_t)

        best_thrs[task] = thrs

    thr_str = {t: ["%.3f" % x for x in v] for t, v in best_thrs.items()}
    print(f"    Calibrated thresholds: {thr_str}")
    return best_thrs


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 — Temperature scaling  (post-hoc probability calibration)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_temperature(model, val_loader, device, config) -> float:
    """
    Optimise a single temperature scalar T on the validation set by minimising
    ordinal BCE NLL.  Run with torch.enable_grad() inside to allow LBFGS.

    After training with ASL (which sharpens probabilities), P(Y>=k) can be
    overconfident.  Temperature > 1.0 softens them.
    """
    model.eval()
    all_logits = {"mgi": [], "ohi": [], "gei": []}
    all_labels = {"mgi": [], "ohi": [], "gei": []}

    with torch.no_grad():
        for batch in val_loader:
            f, l, r, targets = _collect_batch(batch, device)
            out = model(f, l, r)
            for task in ("mgi", "ohi", "gei"):
                all_logits[task].append(out[task].detach().cpu())
                all_labels[task].extend(targets[task].cpu().numpy().tolist())

    cat_logits = {t: torch.cat(all_logits[t], 0) for t in ("mgi", "ohi", "gei")}
    nc_map     = {"mgi": config["num_classes_mgi"],
                  "ohi": config["num_classes_ohi"],
                  "gei": config["num_classes_gei"]}

    def _bce_t(lbls, K):
        y = torch.tensor(lbls, dtype=torch.long)
        B = len(y); t = torch.zeros(B, K - 1)
        for k in range(1, K):
            t[:, k - 1] = (y >= k).float()
        return t

    bce_tgts = {t: _bce_t(all_labels[t], nc_map[t]) for t in nc_map}

    T = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=config["temp_lr"],
                              max_iter=config["temp_max_iter"])

    def _nll():
        opt.zero_grad()
        loss = sum(
            F.binary_cross_entropy_with_logits(
                cat_logits[t] / T.clamp(min=0.05), bce_tgts[t])
            for t in nc_map
        )
        loss.backward()
        return loss

    # Need gradients for temperature parameter
    with torch.enable_grad():
        opt.step(_nll)

    T_val = float(T.detach().clamp(0.5, 5.0).item())
    print(f"    Temperature = {T_val:.4f}")
    return T_val


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 — EMA + SWA helpers
# ─────────────────────────────────────────────────────────────────────────────

class EMAHelper:
    """CPU-side exponential moving average of model weights."""

    def __init__(self, decay: float = 0.998):
        self.decay       = decay
        self.shadow      = {}
        self.initialized = False

    def register(self, model: nn.Module):
        self.shadow      = {k: v.detach().float().cpu().clone()
                            for k, v in model.state_dict().items()
                            if torch.is_floating_point(v)}
        self.initialized = True

    def update(self, model: nn.Module):
        if not self.initialized:
            self.register(model); return
        d = self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if not torch.is_floating_point(v): continue
                cur = v.detach().float().cpu()
                if k not in self.shadow:
                    self.shadow[k] = cur.clone()
                else:
                    self.shadow[k].mul_(d).add_(cur, alpha=1.0 - d)

    def apply_to(self, model: nn.Module) -> dict:
        """Swap EMA weights into model; return backup of original weights."""
        if not self.initialized: return {}
        backup = {}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow and torch.is_floating_point(v):
                    backup[k] = v.detach().clone()
                    v.copy_(self.shadow[k].to(device=v.device, dtype=v.dtype))
        return backup

    def restore(self, model: nn.Module, backup: dict):
        if not backup: return
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in backup:
                    v.copy_(backup[k].to(device=v.device, dtype=v.dtype))


class SWAHelper:
    """Uniform running average of model weight snapshots."""

    def __init__(self):
        self.n      = 0
        self.shadow = None

    def update(self, model: nn.Module):
        sd = {k: v.detach().cpu().float().clone()
              for k, v in model.state_dict().items()}
        if self.shadow is None:
            self.shadow = sd
        else:
            n = self.n
            self.shadow = {
                k: self.shadow[k] * (n / (n + 1)) + sd[k] * (1.0 / (n + 1))
                for k in sd}
        self.n += 1

    def apply_to(self, model: nn.Module):
        if self.shadow is None: return
        cur = model.state_dict()
        model.load_state_dict({
            k: self.shadow[k].to(device=cur[k].device, dtype=cur[k].dtype)
            for k in self.shadow})
        log.info(f"SWA applied ({self.n} snapshots).")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17 — Safe file I/O (Windows file-lock tolerant)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_save(payload, path, tag="ckpt") -> str:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        tmp = dst.with_suffix(f"{dst.suffix}.tmp.{os.getpid()}.{attempt}")
        try:
            torch.save(payload, tmp)
            os.replace(tmp, dst)
            return str(dst)
        except OSError:
            try: tmp.unlink()
            except: pass
            time.sleep(0.6 * (attempt + 1))
    # Fallback to timestamped name
    fb = dst.with_name(f"{dst.stem}_{int(time.time())}{dst.suffix}")
    torch.save(payload, fb)
    log.warning(f"{tag}: locked; saved fallback → {fb}")
    return str(fb)


def _safe_json(payload, path) -> str:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        tmp = dst.with_suffix(f"{dst.suffix}.tmp.{os.getpid()}.{attempt}")
        try:
            with open(tmp, "w") as fp:
                json.dump(payload, fp, indent=2)
            os.replace(tmp, dst)
            return str(dst)
        except OSError:
            try: tmp.unlink()
            except: pass
            time.sleep(0.5 * (attempt + 1))
    fb = dst.with_name(f"{dst.stem}_{int(time.time())}{dst.suffix}")
    with open(fb, "w") as fp:
        json.dump(payload, fp, indent=2)
    return str(fb)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 18 — K-Fold training main loop
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.freeze_support()
    seed_everything(CONFIG["seed"])

    if not torch.cuda.is_available():
        raise RuntimeError(
            "\nCUDA GPU required.\n"
            "Install: pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu126 --force-reinstall\n"
            "Verify:  python -c \"import torch; print(torch.cuda.is_available())\""
        )

    DEVICE = torch.device("cuda")
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Torch: {torch.__version__}  |  timm: {timm.__version__}")

    # Speed flags (non-deterministic but faster)
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        try: torch.set_float32_matmul_precision("high")
        except: pass

    for d in (CONFIG["checkpoint_dir"], CONFIG["plots_dir"]):
        os.makedirs(d, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    records, patient_labels = parse_thesis_csv(CONFIG["data_dir"])
    print("\n── LABEL DISTRIBUTION ───────────────────────────────────────────")
    for tag, key, nc in [("MGI(0-4)", "mgi", 5),
                          ("OHI(0-2)", "ohi", 3),
                          ("GEI(0-2)", "gei", 3)]:
        vals   = [v[key] for v in patient_labels.values()]
        counts = np.bincount(vals, minlength=nc)
        line   = "  ".join(f"c{i}:{n}({n/len(vals)*100:.0f}%)"
                            + (" ⚠" if n < 15 else "")
                            for i, n in enumerate(counts))
        print(f"  {tag}: {line}")
    print("─────────────────────────────────────────────────────────────────\n")

    POS_WEIGHTS   = compute_pos_weights(patient_labels, CONFIG)
    FULL_TRAIN_DS = MultiViewPatientDataset(
        records, TRAIN_AUG_NORMAL, TRAIN_AUG_STRONG)
    FULL_VAL_DS   = MultiViewPatientDataset(records, VAL_AUG)

    NC_MAP        = {"mgi": CONFIG["num_classes_mgi"],
                     "ohi": CONFIG["num_classes_ohi"],
                     "gei": CONFIG["num_classes_gei"]}
    all_mgi       = np.array(FULL_TRAIN_DS.mgi_labels)
    n_patients    = len(FULL_TRAIN_DS)

    # Determine safe fold count (limited by rarest class)
    mgi_counts = np.bincount(all_mgi, minlength=5)
    eff_k      = min(CONFIG["k_folds"], int(mgi_counts[mgi_counts > 0].min()))
    if eff_k < 2:
        raise RuntimeError(f"Too few samples: MGI counts {mgi_counts}")
    if eff_k != CONFIG["k_folds"]:
        log.warning(f"k_folds reduced {CONFIG['k_folds']} → {eff_k}")

    skf = StratifiedKFold(n_splits=eff_k, shuffle=True,
                          random_state=CONFIG["seed"])

    fold_summaries = []
    all_swa_ckpts  = []
    all_histories  = []

    print(f"\n{'='*72}")
    print(f"  DentAI v4  |  {eff_k} folds  ×  {CONFIG['num_epochs']} epochs")
    print(f"  DINOv2 CLS+patch (768d) + EfficientNet-B4 (1792d) → 256d")
    print(f"  Loss: OrdBCE+ASL+MAE+Mono+SupCon(per-task)+GEI-aux+R-Drop")
    print(f"  New: CutMix | Per-threshold calib | Temp scaling | All-fold ensemble")
    print(f"{'='*72}\n")

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.arange(n_patients), all_mgi), start=1):
        seed_everything(CONFIG["seed"] + fold_idx)
        print(f"\n{'─'*72}")
        print(f"  FOLD {fold_idx}/{eff_k}   "
              f"train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'─'*72}")

        base_train  = list(train_idx)
        replay_pool = build_hard_replay(FULL_TRAIN_DS, base_train)
        print(f"  Hard replay pool: {len(replay_pool)}/{len(base_train)}")

        val_loader = DataLoader(
            Subset(FULL_VAL_DS, list(val_idx)),
            batch_size=CONFIG["batch_size"], shuffle=False,
            num_workers=CONFIG["num_workers"])

        model   = OralHealthModelV4(CONFIG).to(DEVICE)
        loss_fn = MultiTaskLossV4(POS_WEIGHTS, CONFIG).to(DEVICE)
        scaler  = _make_scaler()
        swa     = SWAHelper()
        ema     = EMAHelper(CONFIG["ema_decay"])

        # Phase 1: freeze backbone, train heads only
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            model.head_parameters(),
            lr=CONFIG["lr_heads"],
            weight_decay=CONFIG["weight_decay"])
        sched_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["freeze_epochs"],
            eta_min=CONFIG["lr_heads"] * 0.1)

        ckpt_path   = Path(CONFIG["checkpoint_dir"]) / f"fold_{fold_idx}_best.pth"
        swa_path    = Path(CONFIG["checkpoint_dir"]) / f"fold_{fold_idx}_swa.pth"
        saved_ckpt  = None
        best_obj    = -999.0
        patience    = 0
        fold_hist   = []
        phase       = 1

        # Default per-threshold thresholds
        def_thrs = {t: [0.5] * (NC_MAP[t] - 1) for t in NC_MAP}
        best_thrs   = {t: list(def_thrs[t]) for t in NC_MAP}
        temperature = 1.0

        ema_start    = CONFIG["ema_start_epoch"]
        use_ema_eval = CONFIG["use_ema_eval"]

        # ── Epoch loop ─────────────────────────────────────────────────────
        for epoch in range(1, CONFIG["num_epochs"] + 1):

            # Transition: unfreeze backbone at end of freeze phase
            if epoch == CONFIG["freeze_epochs"] + 1:
                phase = 2
                model.unfreeze_last_n_blocks(CONFIG["unfreeze_blocks"])
                remaining = CONFIG["num_epochs"] - CONFIG["freeze_epochs"]

                param_groups = [
                    {"params": model.backbone_parameters(),
                     "lr": CONFIG["lr_backbone"]},
                    {"params": model.dino_pool.parameters(),
                     "lr": CONFIG["lr_projection"]},
                    {"params": model.proj.parameters(),
                     "lr": CONFIG["lr_projection"]},
                    {"params": model.mgi_head.parameters(),
                     "lr": CONFIG["lr_heads"]},
                    {"params": model.ohi_head.parameters(),
                     "lr": CONFIG["lr_heads"]},
                    {"params": model.gei_head.parameters(),
                     "lr": CONFIG["lr_heads"]},
                    {"params": model.gei_aux.parameters(),
                     "lr": CONFIG["lr_heads"]},
                ] + [
                    {"params": getattr(model, f"sc_proj_{t}").parameters(),
                     "lr": CONFIG["lr_projection"]}
                    for t in ("mgi", "ohi", "gei")
                ]
                if CONFIG["cnn_cross_view_attn"]:
                    param_groups.append(
                        {"params": model.cnn_pool.parameters(),
                         "lr": CONFIG["lr_projection"]})

                optimizer = torch.optim.AdamW(
                    param_groups, weight_decay=CONFIG["weight_decay"])

                # SGDR with warm restarts (more exploration than plain cosine)
                sched_p2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0      = CONFIG["sgdr_T0"],
                    T_mult   = CONFIG["sgdr_T_mult"],
                    eta_min  = CONFIG["lr_heads"] * CONFIG["sgdr_eta_min_frac"])

            # Build epoch's training loader with optional hard replay
            replay_frac = (CONFIG["hard_replay_frac"]
                           if epoch >= CONFIG["hard_replay_start_epoch"] else 0.0)
            ep_idx  = make_epoch_indices(base_train, replay_pool, replay_frac)
            sampler = build_sampler(FULL_TRAIN_DS, ep_idx)
            t_loader = DataLoader(
                Subset(FULL_TRAIN_DS, ep_idx),
                batch_size=CONFIG["batch_size"], sampler=sampler,
                drop_last=True, num_workers=CONFIG["num_workers"])

            ema_for_step = (ema
                            if (phase == 2 and epoch >= ema_start) else None)

            # ── Train ───────────────────────────────────────────────────────
            train_loss = run_train_epoch(
                model, t_loader, optimizer, loss_fn, scaler,
                DEVICE, CONFIG, phase, epoch, ema=ema_for_step)

            if phase == 1: sched_p1.step()
            else:          sched_p2.step()

            if epoch >= CONFIG["swa_start_epoch"]:
                swa.update(model)

            # ── Calibration every N epochs ──────────────────────────────────
            ema_active = (phase == 2 and epoch >= ema_start
                          and use_ema_eval and ema.initialized)

            if epoch >= 15 and epoch % CONFIG["calib_every_epochs"] == 0:
                print(f"  [F{fold_idx}] Ep{epoch}: calibrating thresholds...")
                if ema_active:
                    bk = ema.apply_to(model)
                    try:
                        best_thrs = calibrate_per_threshold(
                            model, val_loader, DEVICE, CONFIG, temperature)
                    finally:
                        ema.restore(model, bk)
                else:
                    best_thrs = calibrate_per_threshold(
                        model, val_loader, DEVICE, CONFIG, temperature)

            # ── Validate ────────────────────────────────────────────────────
            if ema_active:
                bk = ema.apply_to(model)
                try:
                    val_loss, yt, yp = run_val_epoch(
                        model, val_loader, loss_fn, DEVICE, CONFIG,
                        best_thrs, temperature)
                finally:
                    ema.restore(model, bk)
            else:
                val_loss, yt, yp = run_val_epoch(
                    model, val_loader, loss_fn, DEVICE, CONFIG,
                    best_thrs, temperature)

            metrics = compute_metrics(yt, yp, NC_MAP)
            obj     = ckpt_objective(metrics, NC_MAP, CONFIG["ckpt_mae_penalty"])
            avg_f1  = float(np.mean([metrics[t]["f1"] for t in NC_MAP]))

            fold_hist.append({
                "epoch": epoch,
                "train_loss": train_loss["total"],
                "val_loss":   val_loss,
                "val_avg_f1": avg_f1,
                "f1_mgi":     metrics["mgi"]["f1"],
                "f1_ohi":     metrics["ohi"]["f1"],
                "f1_gei":     metrics["gei"]["f1"],
                "mae_mgi":    metrics["mgi"]["mae"],
                "objective":  obj,
            })

            if epoch % 5 == 0 or epoch <= 3:
                star = " ★" if obj > best_obj else ""
                em   = "  EMA" if ema_active else ""
                mx   = _mixup_prob(epoch, CONFIG)
                print(f"  Ep{epoch:3d}  "
                      f"tr={train_loss['total']:.3f}  "
                      f"vl={val_loss:.3f}  "
                      f"F1(mgi={metrics['mgi']['f1']:.3f} "
                      f"ohi={metrics['ohi']['f1']:.3f} "
                      f"gei={metrics['gei']['f1']:.3f})  "
                      f"obj={obj:.4f}  mx={mx:.2f}"
                      f"{em}{star}")

            # ── Best model checkpoint ────────────────────────────────────────
            if obj > best_obj:
                best_obj  = obj
                patience  = 0
                payload   = {
                    "epoch":            epoch,
                    "model_state_dict": None,  # filled below
                    "objective":        obj,
                    "metrics":          metrics,
                    "thresholds":       {t: list(v) for t, v in best_thrs.items()},
                    "temperature":      temperature,
                    "config":           CONFIG,
                }
                if ema_active:
                    bk = ema.apply_to(model)
                    try:
                        payload["model_state_dict"] = {
                            k: v.cpu() for k, v in model.state_dict().items()}
                        saved_ckpt = _safe_save(payload, ckpt_path, f"F{fold_idx}")
                    finally:
                        ema.restore(model, bk)
                else:
                    payload["model_state_dict"] = {
                        k: v.cpu() for k, v in model.state_dict().items()}
                    saved_ckpt = _safe_save(payload, ckpt_path, f"F{fold_idx}")
            else:
                patience += 1

            if patience >= CONFIG["early_stopping_patience"] and phase == 2:
                print(f"  Early stop at epoch {epoch}.")
                break

        # ── Post-fold: temperature scaling + re-calibration ────────────────
        if saved_ckpt is None:
            # Edge case: nothing improved → save current state
            payload = {
                "epoch": epoch,
                "model_state_dict": {k: v.cpu()
                                     for k, v in model.state_dict().items()},
                "objective": best_obj, "metrics": {},
                "thresholds": {t: list(v) for t, v in best_thrs.items()},
                "temperature": temperature, "config": CONFIG,
            }
            saved_ckpt = _safe_save(payload, ckpt_path, f"F{fold_idx}_fb")

        # Load best model for temp-scaling + final calibration
        m_tmp = OralHealthModelV4(CONFIG).to(DEVICE)
        ck    = torch.load(saved_ckpt, map_location=DEVICE, weights_only=False)
        m_tmp.load_state_dict(ck["model_state_dict"])

        print(f"  Temperature scaling (fold {fold_idx})...")
        temperature = optimize_temperature(m_tmp, val_loader, DEVICE, CONFIG)
        print(f"  Final threshold calibration (T={temperature:.4f})...")
        best_thrs   = calibrate_per_threshold(
            m_tmp, val_loader, DEVICE, CONFIG, temperature)

        # Update saved checkpoint with calibration info
        ck["temperature"] = temperature
        ck["thresholds"]  = {t: list(v) for t, v in best_thrs.items()}
        saved_ckpt        = _safe_save(ck, ckpt_path, f"F{fold_idx}_calib")
        del m_tmp; torch.cuda.empty_cache()

        # ── SWA model ───────────────────────────────────────────────────────
        swa_temperature = temperature
        swa_thrs        = {t: list(v) for t, v in best_thrs.items()}

        if swa.n > 0:
            m_swa = OralHealthModelV4(CONFIG).to(DEVICE)
            ck_   = torch.load(saved_ckpt, map_location=DEVICE, weights_only=False)
            m_swa.load_state_dict(ck_["model_state_dict"])
            swa.apply_to(m_swa)

            print(f"  SWA temp scaling + calibration ({swa.n} snapshots)...")
            swa_temperature = optimize_temperature(
                m_swa, val_loader, DEVICE, CONFIG)
            swa_thrs = calibrate_per_threshold(
                m_swa, val_loader, DEVICE, CONFIG, swa_temperature)

            swa_payload = {
                "model_state_dict": {k: v.cpu()
                                     for k, v in m_swa.state_dict().items()},
                "swa_n":        swa.n,
                "thresholds":   {t: list(v) for t, v in swa_thrs.items()},
                "temperature":  swa_temperature,
                "config":       CONFIG,
            }
            saved_swa = _safe_save(swa_payload, swa_path, f"F{fold_idx}_swa")
            del m_swa; torch.cuda.empty_cache()
        else:
            # SWA window not reached: copy best checkpoint as SWA fallback
            shutil.copy(saved_ckpt, swa_path)
            saved_swa = str(swa_path)
            log.info(f"SWA not triggered (< {CONFIG['swa_start_epoch']} epochs); "
                     f"using best ckpt as SWA.")

        all_swa_ckpts.append(saved_swa)

        # ── Fold summary ────────────────────────────────────────────────────
        beh = max(fold_hist, key=lambda h: h["objective"])
        fold_summaries.append({
            "fold":           fold_idx,
            "best_obj":       best_obj,
            "best_epoch":     beh["epoch"],
            "f1_mgi":         beh["f1_mgi"],
            "f1_ohi":         beh["f1_ohi"],
            "f1_gei":         beh["f1_gei"],
            "mae_mgi":        beh["mae_mgi"],
            "temperature":    temperature,
            "swa_temperature": swa_temperature,
            "thresholds":     {t: list(v) for t, v in best_thrs.items()},
            "swa_thresholds": {t: list(v) for t, v in swa_thrs.items()},
            "checkpoint":     str(saved_ckpt),
            "swa_checkpoint": saved_swa,
        })
        all_histories.append(fold_hist)

        print(f"\n  Fold {fold_idx}: obj={best_obj:.4f}  "
              f"mgi={beh['f1_mgi']:.3f}  "
              f"ohi={beh['f1_ohi']:.3f}  "
              f"gei={beh['f1_gei']:.3f}  T={temperature:.4f}")

        del model, loss_fn, optimizer, scaler, swa, ema
        torch.cuda.empty_cache()


    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 19 — Cross-fold results summary
    # ─────────────────────────────────────────────────────────────────────────

    df_sum = pd.DataFrame(fold_summaries)
    print(f"\n{'='*72}")
    print("  FOLD RESULTS")
    print(df_sum[["fold", "best_obj", "best_epoch",
                  "f1_mgi", "f1_ohi", "f1_gei", "mae_mgi"]].to_string(index=False))
    print("\n  Cross-fold mean ± std:")
    for c in ["f1_mgi", "f1_ohi", "f1_gei"]:
        v = df_sum[c]
        print(f"    {c:8s}: {v.mean():.4f} ± {v.std():.4f}")
    print(f"{'='*72}\n")


    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 20 — All-fold SWA ensemble evaluation
    # ─────────────────────────────────────────────────────────────────────────

    # Average thresholds and temperatures across all folds
    avg_thrs = {}
    for task in NC_MAP:
        all_t = [fold_summaries[i]["swa_thresholds"][task]
                 for i in range(len(fold_summaries))]
        avg_thrs[task] = list(np.mean(all_t, axis=0))
    avg_temp = float(np.mean([s["swa_temperature"] for s in fold_summaries]))

    # Use best fold's val set for ensemble demo (least optimistic)
    best_fi = int(df_sum["best_obj"].idxmax())
    _, val_best = list(skf.split(np.arange(n_patients), all_mgi))[best_fi]
    vl_best = DataLoader(Subset(FULL_VAL_DS, list(val_best)),
                          batch_size=1, shuffle=False,
                          num_workers=CONFIG["num_workers"])

    print(f"Loading {len(all_swa_ckpts)} SWA fold models for ensemble eval...")
    ens_models = []
    ens_thrs   = []
    ens_temps  = []
    for ckpt_p in all_swa_ckpts:
        m = OralHealthModelV4(CONFIG).to(DEVICE)
        try:    ck = torch.load(ckpt_p, map_location=DEVICE, weights_only=False)
        except: ck = torch.load(ckpt_p, map_location=DEVICE)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        ens_models.append(m)
        ens_thrs.append({"mgi": avg_thrs["mgi"],
                          "ohi": avg_thrs["ohi"],
                          "gei": avg_thrs["gei"]})
        ens_temps.append(float(ck.get("temperature", avg_temp)))

    yt_ens = {"mgi": [], "ohi": [], "gei": []}
    yp_ens = {"mgi": [], "ohi": [], "gei": []}

    with torch.no_grad():
        for batch in vl_best:
            f, l, r, labels = batch
            f = f.to(DEVICE); l = l.to(DEVICE); r = r.to(DEVICE)

            preds = ensemble_tta_predict(
                ens_models, f, l, r,
                ens_thrs, ens_temps,
                n_tta=CONFIG["tta_steps"])

            for task in ("mgi", "ohi", "gei"):
                yt_ens[task].extend(labels[task].numpy().tolist())
                yp_ens[task].extend(preds[task].cpu().numpy().tolist())

    print("\n── ALL-FOLD SWA ENSEMBLE  (TTA × 7) ────────────────────────────")
    for task, nc in NC_MAP.items():
        f1 = f1_score(yt_ens[task], yp_ens[task],
                       average="macro", zero_division=0)
        print(f"\n  {task.upper()} macro-F1: {f1:.4f}")
        print(classification_report(yt_ens[task], yp_ens[task],
                                     zero_division=0))

    for m in ens_models:
        del m
    torch.cuda.empty_cache()


    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 21 — Plots
    # ─────────────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    for fi, hist in enumerate(all_histories):
        ep = [h["epoch"] for h in hist]
        axes[0, 0].plot(ep, [h["train_loss"] for h in hist],
                        label=f"F{fi+1}", alpha=0.8)
        axes[0, 1].plot(ep, [h["val_loss"]   for h in hist],
                        label=f"F{fi+1}", alpha=0.8)
        axes[0, 2].plot(ep, [h["val_avg_f1"] for h in hist],
                        label=f"F{fi+1}", alpha=0.8)
    for ax, t in zip(axes[0], ["Train Loss", "Val Loss", "Val Avg Macro-F1"]):
        ax.set_title(t, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    task_plot_cfg = [
        (f"MGI 0-{NC_MAP['mgi']-1}", "mgi", NC_MAP["mgi"]),
        (f"OHI 0-{NC_MAP['ohi']-1} merged", "ohi", NC_MAP["ohi"]),
        (f"GEI 0-{NC_MAP['gei']-1} merged", "gei", NC_MAP["gei"]),
    ]
    for ax, (title, task, nc) in zip(axes[1], task_plot_cfg):
        cm     = confusion_matrix(yt_ens[task], yp_ens[task],
                                   labels=list(range(nc)))
        cmnorm = (cm.astype(float) /
                  cm.sum(axis=1, keepdims=True).clip(min=1))
        sns.heatmap(cmnorm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                    xticklabels=range(nc), yticklabels=range(nc))
        ax.set_title(f"{title} — Ensemble Confusion", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.suptitle("DentAI v4 — Training Results (All-fold SWA Ensemble)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(CONFIG["plots_dir"], "training_results_v4.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot → {plot_path}")


    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 22 — Export ensemble config for Django inference backend
    # ─────────────────────────────────────────────────────────────────────────

    avg_f1 = {t: float(df_sum[f"f1_{t}"].mean()) for t in ("mgi", "ohi", "gei")}

    ensemble_cfg = {
        "model_type":       "OralHealthModelV4",
        "models":            all_swa_ckpts,
        "avg_thresholds":    {t: list(v) for t, v in avg_thrs.items()},
        "avg_temperature":   avg_temp,
        "avg_val_f1":        avg_f1,
        "config":            CONFIG,
        "label_mapping": {
            "mgi": "0,1,2,3,4 (unchanged)",
            "ohi": "0,1,2 — original 2+3 merged into 2",
            "gei": "0,1,2 — original 2+3 merged into 2",
        },
        "decode_instruction": (
            "probs = sigmoid(logits / temperature);  "
            "predicted_class = sum(probs > per_threshold_list)"
        ),
        "inference_note": (
            "Load all models in 'models' list, run TTA on each, "
            "average sigmoid probs, then decode with avg_thresholds."
        ),
    }

    for sp in [
        Path(CONFIG["checkpoint_dir"]) / "ensemble_config.json",
        Path(CONFIG["checkpoint_dir"]).parent / "ensemble_config.json",
    ]:
        out_path = _safe_json(ensemble_cfg, sp)
        print(f"Ensemble config → {out_path}")

    print(f"\n{'='*72}")
    print(f"  DONE — DentAI v4")
    print(f"  Mean val F1:  MGI={avg_f1['mgi']:.4f}  "
          f"OHI={avg_f1['ohi']:.4f}  GEI={avg_f1['gei']:.4f}")
    print(f"  Avg temperature:  {avg_temp:.4f}")
    print(f"  Per-threshold decoding: "
          f"MGI={['%.3f'%x for x in avg_thrs['mgi']]}  "
          f"OHI={['%.3f'%x for x in avg_thrs['ohi']]}  "
          f"GEI={['%.3f'%x for x in avg_thrs['gei']]}")
    print(f"{'='*72}\n")


    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 23 — Smoke test (forward pass + decode verification)
    # ─────────────────────────────────────────────────────────────────────────

    print("── Smoke test ───────────────────────────────────────────────────")
    m_test = OralHealthModelV4(CONFIG).to("cpu")
    try:
        ck_t = torch.load(all_swa_ckpts[0], map_location="cpu", weights_only=False)
    except TypeError:
        ck_t = torch.load(all_swa_ckpts[0], map_location="cpu")
    m_test.load_state_dict(ck_t["model_state_dict"])
    m_test.eval()

    with torch.no_grad():
        dummy = torch.zeros(1, 3, CONFIG["image_size"], CONFIG["image_size"])
        out_t = m_test(dummy, dummy, dummy, return_features=True)

    print("✓ Forward pass OK")
    print(f"  Task keys:     {[k for k in out_t if k != 'features']}")
    print(f"  Feature tasks: {list(out_t['features'].keys())}")
    print(f"  Output shapes:")
    for k in ("mgi", "ohi", "gei"):
        print(f"    {k}: {tuple(out_t[k].shape)}  "
              f"(K-1={NC_MAP[k]-1} thresholds)")

    thr_t  = ck_t.get("thresholds",  {t: [0.5]*(NC_MAP[t]-1) for t in NC_MAP})
    temp_t = float(ck_t.get("temperature", 1.0))
    decoded = {k: decode_ordinal(out_t[k], thr_t.get(k, 0.5), temp_t).item()
               for k in ("mgi", "ohi", "gei")}
    print(f"  Decoded (zero input): {decoded}")
    del m_test

    print("─────────────────────────────────────────────────────────────────")
    print(f"Checkpoints: {CONFIG['checkpoint_dir']}")
    print("Done.")
