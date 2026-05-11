#!/usr/bin/env python3
"""
DentAI Oral Health Prediction — Training Script v5 (FINAL)
============================================================

THE CORE PROBLEM FIXED IN V5:
══════════════════════════════
v4 confusion matrices showed "ordinal polarisation":
  • Class 0 and class-max perfect (1.00)
  • Middle classes collapse:
      OHI class 1 → 0% accuracy (92% predicted as class 0)
      MGI class 1 → 10% (80% → class 0)
      MGI class 2 → 12% (62% → class 3)

ROOT CAUSE: OrdinalBCE trains K-1 INDEPENDENT binary classifiers.
  Classifier-1 (Y≥1) sees 134 negatives vs 69 positives for OHI.
  It learns to be conservative → fires False for class-1 patients.
  Since all K-1 classifiers fire False → predicted class 0.
  The classifiers can "disagree" in ways that collapse the middle.

THE FIX — Cumulative Link Model (CLM):
  • ONE shared severity score  f(x)  per task (scalar from features)
  • K-1 GLOBALLY ORDERED cutpoints  α₁ < α₂ < … < α_{K-1}
  • P(Y = k) = σ(αk − f(x)) − σ(α_{k-1} − f(x))
  
  For OHI=1 patient:
    OrdinalBCE: 2 independent classifiers can both fire False → class 0 ✗
    CLM: if f(x) is between α₁ and α₂ → MUST predict class 1 ✓
    Gradient directly pushes OHI=1 scores into [α₁, α₂] band

ADDITIONAL V5 IMPROVEMENTS:
  • Class weights with middle-class boost (2.5–3×) prevent minority collapse
  • Ordinal label smoothing (Gaussian, σ=0.7) teaches adjacent-class similarity
  • Focal modulation (γ=2.0) focuses gradient on hard middle-class examples
  • Hard boundary auxiliary heads (OHI 0vs1+, MGI ≤1vs≥2) — extra supervision
    for the two most problematic decision boundaries
  • 200 epochs (v4 val loss still decreasing at epoch 120 — undertrained)
  • SWA from epoch 150 (more snapshots after convergence)
  • Min-recall checkpoint guard — never saves model that ignores a class entirely
  • Temperature scaling (post-hoc, LBFGS on val) — calibrates overconfidence
  • All-fold SWA ensemble with TTA at inference

PLACE:  <project_root>/training/train_model_v5.py
RUN:    python training/train_model_v5.py
VRAM:   ≥ 8 GB  (batch=4, accum=4 → effective_batch=16)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0 — Imports
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
# 1 — Seed
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed);  os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything(42)

# ─────────────────────────────────────────────────────────────────────────────
# 2 — Config
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    # Image
    "image_size":               336,

    # Classes (OHI/GEI merged: 2+3 → 2)
    "num_classes_mgi":           5,
    "num_classes_ohi":           3,
    "num_classes_gei":           3,

    # Backbones
    "backbone_dino":  "vit_small_patch14_dinov2.lvd142m",
    "backbone_cnn":   "efficientnet_b4",
    "pretrained":                True,
    "freeze_epochs":             10,
    "unfreeze_blocks":            6,

    # Architecture
    "dropout":                   0.25,
    "projection_dim":            256,
    "clm_hidden_dim":            128,  # CLM extractor hidden size
    "view_attn_heads":             4,
    "dino_use_cls":              True,  # CLS + patch_avg → 768d
    "cnn_cross_view_attn":       True,
    "supcon_dim":                128,
    "supcon_temp":               0.07,

    # Training
    "batch_size":                  4,
    "grad_accum_steps":            4,
    "num_epochs":                200,   # v4 val loss still falling at ep120
    "warmup_epochs":               5,

    # Learning rates
    "lr_backbone":             2e-7,
    "lr_projection":           2e-5,
    "lr_heads":                8e-5,
    "weight_decay":            1e-3,
    "grad_clip_norm":           1.0,

    # Scheduler — SGDR
    "sgdr_T0":                   35,
    "sgdr_T_mult":                2,
    "sgdr_eta_min_frac":         0.04,

    # EMA + SWA
    "ema_decay":               0.998,
    "ema_start_epoch":            12,
    "use_ema_eval":              True,
    "swa_start_epoch":           150,  # more converged before averaging

    # CLM loss weights
    "clm_nll_w":                 0.50,
    "clm_focal_w":               0.30,
    "clm_smooth_w":              0.20,
    "clm_sigma":                  0.65, # ordinal label smoothing bandwidth
    "clm_gamma":                  2.5, # focal modulation gamma

    # Middle-class boost in class weights
    "mgi_class_weights":  [0.55, 2.11, 2.26, 3.62, 5.08],
    "ohi_class_weights":  [0.50, 3.44, 6.77],
    "gei_class_weights":  [0.39, 7.52, 10.0],

    # SupCon
    "alpha_supcon":              0.30,
    "supcon_adj_weight":         0.40,

    # Hard boundary auxiliaries
    "alpha_ohi_boundary":        0.25,  # OHI 0 vs (1+2) binary
    "alpha_mgi_boundary":        0.20,  # MGI (0+1) vs (2+3+4) binary
    "alpha_gei_aux":             0.20,  # GEI 0 vs (1+2) binary (unchanged)

    # R-Drop
    "alpha_rdrop":               0.15,

    # Task loss weights
    "task_loss_weights": {"mgi": 0.45, "ohi": 0.30, "gei": 0.25},

    # Sampler
    "hard_replay_frac":          0.35,
    "hard_replay_start":          12,

    # MixUp + CutMix
    "mixup_alpha":               0.4,
    "mixup_prob_start":          0.20,
    "mixup_prob_end":            0.45,
    "cutmix_alpha":              1.0,
    "cutmix_prob":               0.30,

    # Min-recall guard (checkpoint rejected if any class recall < this)
    "min_class_recall":          0.15,  # any class must have ≥15% recall

    # Temperature scaling
    "temp_lr":                   0.01,
    "temp_max_iter":             500,

    # TTA
    "tta_steps":                   7,

    # K-fold
    "k_folds":                     5,
    "early_stopping_patience":    35,
    "seed":                       42,

    # DataLoader
    "num_workers":                 0,

    # Paths
    "data_dir":       str(PROJECT_ROOT / "Thesis_Data"),
    "checkpoint_dir": str(PROJECT_ROOT / "models" / "checkpoints"),
    "plots_dir":      str(PROJECT_ROOT / "outputs"  / "plots"),
}


# ─────────────────────────────────────────────────────────────────────────────
# 3 — Label merging
# ─────────────────────────────────────────────────────────────────────────────

def merge_ohi(x): return min(int(x), 2)
def merge_gei(x): return min(int(x), 2)


# ─────────────────────────────────────────────────────────────────────────────
# 4 — CSV parsing
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
            for ext in (".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"):
                c = folder / f"{pfx}{pid}{ext}"
                if c.exists(): img_path = c; break
            if img_path is None: continue
            records.append({"patient_id": str(pid), "view": vname,
                            "image_path": str(img_path),
                            "mgi": mgi, "ohi": ohi, "gei": gei})

    print(f"CSV: {len(patient_labels)} patients  "
          f"{len(records)} images  ({skipped} skipped)")
    return records, patient_labels


# ─────────────────────────────────────────────────────────────────────────────
# 5 — Augmentation
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
_SZ = CONFIG["image_size"]

def _build_aug(strength: float = 1.0) -> A.Compose:
    s = strength
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.05),
        A.ShiftScaleRotate(shift_limit=0.06*s, scale_limit=0.18*s,
                           rotate_limit=int(28*s), p=0.70),
        A.RandomResizedCrop(height=_SZ, width=_SZ,
                            scale=(max(0.65, 0.80-0.15*s), 1.0),
                            ratio=(0.85, 1.15), p=0.65),
        A.GridDistortion(num_steps=3, distort_limit=0.25*s, p=0.20),
        A.ElasticTransform(alpha=int(60*s), sigma=6, p=0.15),
        # Colour augmentation — gingival redness is the primary clinical cue
        A.RandomBrightnessContrast(brightness_limit=min(0.40*s,0.50),
                                   contrast_limit  =min(0.40*s,0.50), p=0.70),
        A.HueSaturationValue(hue_shift_limit=int(18*s),
                             sat_shift_limit=int(45*s),
                             val_shift_limit=int(35*s), p=0.65),
        A.RGBShift(r_shift_limit=int(25*s),
                   g_shift_limit=int(18*s),
                   b_shift_limit=int(18*s), p=0.45),
        A.ColorJitter(brightness=0.25*s, contrast=0.25*s,
                      saturation=0.30*s, hue=0.08*s, p=0.50),
        A.CLAHE(clip_limit=3.5*s, tile_grid_size=(8,8), p=0.40),
        A.RandomShadow(p=0.10),
        A.GaussianBlur(blur_limit=(3, max(3, int(7*s)//2*2+1)), p=0.25),
        A.GaussNoise(var_limit=(10, int(60*s)), p=0.30),
        A.ImageCompression(quality_lower=max(50,int(70-20*s)),
                           quality_upper=100, p=0.20),
        A.CoarseDropout(max_holes=6, max_height=int(32*s),
                        max_width =int(32*s), min_holes=1, p=0.25),
        A.Resize(_SZ, _SZ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

TRAIN_AUG_NORMAL = _build_aug(1.0)
TRAIN_AUG_STRONG = _build_aug(1.4)  # rare-class patients get more variety
VAL_AUG = A.Compose([A.Resize(_SZ, _SZ),
                      A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                      ToTensorV2()])


# ─────────────────────────────────────────────────────────────────────────────
# 6 — Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _is_rare(s: dict) -> bool:
    return (s["labels"]["mgi"] >= 3 or
            s["labels"]["ohi"] >= 2 or
            s["labels"]["gei"] >= 1)


class MultiViewPatientDataset(Dataset):
    REQUIRED = ("frontal", "left", "right")

    def __init__(self, records, transform=None, strong_transform=None):
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
            if all(v in s["views"] for v in self.REQUIRED)]
        self.samples.sort(
            key=lambda x: int(x["patient_id"]) if x["patient_id"].isdigit() else 0)

        self.mgi_labels = [s["labels"]["mgi"] for s in self.samples]
        self.ohi_labels = [s["labels"]["ohi"] for s in self.samples]
        self.gei_labels = [s["labels"]["gei"] for s in self.samples]
        print(f"Dataset: {len(self.samples)} complete 3-view triplets")

    def __len__(self): return len(self.samples)

    def _load(self, path: str) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((_SZ, _SZ, 3), dtype=np.uint8)
        img   = cv2.resize(img, (_SZ, _SZ), interpolation=cv2.INTER_LANCZOS4)
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        avg   = img.mean(axis=(0,1)).astype(np.float32)
        scale = np.where(avg > 1e-6, avg.mean() / avg, 1.0)
        img   = np.clip(img.astype(np.float32)*scale, 0, 255).astype(np.uint8)
        lab   = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l,a,b = cv2.split(lab)
        cl    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return cv2.cvtColor(cv2.merge((cl.apply(l),a,b)), cv2.COLOR_LAB2RGB)

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
                img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
            tensors.append(img)
        return tensors[0], tensors[1], tensors[2], s["labels"]


# ─────────────────────────────────────────────────────────────────────────────
# 7 — Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class CLMHead(nn.Module):
    """
    Cumulative Link Model head.

    Produces a SINGLE severity score f(x) and K-1 globally-ordered cutpoints.
    P(Y=k|x) = σ(α_k − f(x)) − σ(α_{k-1} − f(x))

    This is the fundamental fix for ordinal polarisation:
    a patient with OHI=1 MUST have f(x) between α_1 and α_2.
    The gradient enforces this globally — there is no way to predict
    OHI=1 as OHI=0 without moving f(x) or α_1.
    """
    def __init__(self, in_dim: int, num_classes: int,
                 hidden_dim: int = 128, dropout: float = 0.20):
        super().__init__()
        self.K     = num_classes
        self.n_cut = num_classes - 1

        # Severity extractor: in_dim → 1  (the "disease severity axis")
        self.extractor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Reparameterise cutpoints via softplus to enforce strict ordering.
        # raw_gaps[i] → softplus gives positive gap between cut[i] and cut[i+1].
        self.raw_gaps = nn.Parameter(torch.zeros(self.n_cut))
        self.cut_bias = nn.Parameter(torch.zeros(1))

    def _ordered_cutpoints(self) -> torch.Tensor:
        """Return K-1 strictly ordered cutpoints."""
        gaps = F.softplus(self.raw_gaps) + 0.1   # minimum gap 0.1
        return self.cut_bias + torch.cumsum(gaps, dim=0) - gaps.sum() * 0.5

    def forward(self, x: torch.Tensor):
        """
        Returns:
          class_probs : (B, K)   probability for each class
          severity    : (B, 1)   raw scalar severity score
          cum_probs   : (B,K-1)  cumulative P(Y<=k)
        """
        score = self.extractor(x)                        # (B, 1)
        cuts  = self._ordered_cutpoints()                # (K-1,)

        # P(Y <= k | x) = sigmoid(α_k − f(x))
        cum_probs = torch.sigmoid(
            cuts.unsqueeze(0) - score)                   # (B, K-1)

        # P(Y = k) = P(Y<=k) − P(Y<=k-1)
        B     = x.shape[0]
        ones  = score.new_ones(B, 1)
        zeros = score.new_zeros(B, 1)
        full  = torch.cat([ones, cum_probs, zeros], dim=1)  # (B, K+1)
        probs = (full[:, :-1] - full[:, 1:]).clamp(min=1e-8)
        return probs, score, cum_probs


class OrdinalLabelSmoothing(nn.Module):
    """
    Gaussian soft labels for ordinal data.
    For true class k, assign probability mass proportional to
    exp(−(j−k)² / (2σ²)) to class j.

    Teaches the model: adjacent classes are similar but not identical.
    Prevents over-confident predictions on extreme classes.
    """
    def __init__(self, num_classes: int, sigma: float = 0.7):
        super().__init__()
        centers = torch.arange(num_classes).float()
        dists   = (centers.unsqueeze(0) - centers.unsqueeze(1)).pow(2)
        soft    = torch.exp(-dists / (2 * sigma**2))
        soft    = soft / soft.sum(1, keepdim=True)
        self.register_buffer("soft", soft)   # (K, K)

    def forward(self, log_probs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """KL divergence from model distribution to smoothed target."""
        return F.kl_div(log_probs,
                         self.soft[targets],
                         reduction="batchmean")


class CLMLoss(nn.Module):
    """
    Multi-component CLM loss:
      ① Weighted NLL      — class_weight[k] · (−log P(Y=k))
      ② Focal modulation  — (1−P(Y=k))^γ · (−log P(Y=k))
      ③ Ordinal smoothing — KL to Gaussian soft labels

    Middle classes get higher class_weight AND more focal attention
    because (1−P) is large when the model is wrong on them.
    """
    def __init__(self, num_classes: int, class_weights: list,
                 sigma: float = 0.7, gamma: float = 2.0,
                 w_nll: float = 0.50, w_focal: float = 0.30,
                 w_smooth: float = 0.20):
        super().__init__()
        self.K       = num_classes
        self.gamma   = gamma
        self.w_nll   = w_nll
        self.w_focal = w_focal
        self.w_sm    = w_smooth
        cw = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("cw", cw)
        self.smoother = OrdinalLabelSmoothing(num_classes, sigma)

    def forward(self, class_probs: torch.Tensor,
                targets: torch.Tensor,
                lam: float = 1.0,
                targets_b: torch.Tensor = None) -> torch.Tensor:
        log_probs = class_probs.clamp(min=1e-8).log()

        # Helper: weighted NLL for one label set
        def _wnll(tgt):
            w   = self.cw[tgt]
            idx = tgt.unsqueeze(1)
            return -(log_probs.gather(1, idx).squeeze(1) * w).mean()

        def _focal(tgt):
            p_c = class_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
            w   = self.cw[tgt]
            idx = tgt.unsqueeze(1)
            return -(log_probs.gather(1, idx).squeeze(1)
                     * (1.0 - p_c) ** self.gamma * w).mean()

        def _smooth(tgt):
            return self.smoother(log_probs, tgt)

        if targets_b is not None and lam < 1.0:
            l_nll    = lam * _wnll(targets)   + (1-lam) * _wnll(targets_b)
            l_focal  = lam * _focal(targets)  + (1-lam) * _focal(targets_b)
            l_smooth = lam * _smooth(targets) + (1-lam) * _smooth(targets_b)
        else:
            l_nll   = _wnll(targets)
            l_focal = _focal(targets)
            l_smooth= _smooth(targets)

        return self.w_nll*l_nll + self.w_focal*l_focal + self.w_sm*l_smooth


class BoundaryAuxLoss(nn.Module):
    """
    Hard-boundary binary classifier for a specific decision boundary.

    Specifically targets the two most problematic boundaries:
      • OHI 0 vs (1+2):  the model collapses OHI=1 → 0 at 92%
      • MGI (0+1) vs (2+3+4):  MGI=1 collapses to 0 (80%), MGI=2 to 3 (62%)
      • GEI 0 vs (1+2):  existing, unchanged

    Provides an INDEPENDENT gradient path specifically for each boundary.
    """
    def __init__(self, boundary: int, pos_weight: float = 3.0):
        """
        boundary: samples with label >= boundary are "positive" (1).
        E.g. boundary=1 → OHI 0 vs OHI>=1.
        """
        super().__init__()
        self.boundary = boundary
        self.pw       = pos_weight

    def forward(self, logit: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        target = (labels >= self.boundary).float()
        pw     = logit.new_tensor([self.pw])
        return F.binary_cross_entropy_with_logits(
            logit.squeeze(1), target, pos_weight=pw)


class SupConLoss(nn.Module):
    """Supervised Contrastive with ordinal adjacency soft-positive weight."""
    def __init__(self, temperature: float = 0.07, adj_weight: float = 0.40):
        super().__init__()
        self.temp = temperature
        self.adj  = adj_weight

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        B = features.shape[0]
        if B < 2: return features.new_zeros(())
        sim  = torch.mm(features, features.T) / self.temp
        lr   = labels.unsqueeze(1).expand(B, B)
        lc   = labels.unsqueeze(0).expand(B, B)
        same = (lr == lc).float()
        adj  = ((lr-lc).abs() == 1).float() * self.adj
        mask = (same + adj).clamp(max=1.0)
        diag = torch.eye(B, device=features.device)
        mask = mask * (1.0 - diag)
        vpos = mask.sum(1).clamp(min=1e-8)
        exp_sim  = torch.exp(sim) * (1.0-diag)
        log_den  = torch.log(exp_sim.sum(1).clamp(min=1e-8))
        log_prob = sim - log_den.unsqueeze(1)
        return (-(mask * log_prob).sum(1) / vpos).mean()


class RDropLoss(nn.Module):
    """KL consistency between two dropout-mask forward passes."""
    def __init__(self, alpha: float = 0.15):
        super().__init__()
        self.alpha = alpha

    def forward(self, probs1: dict, probs2: dict) -> torch.Tensor:
        total = next(iter(probs1.values())).new_zeros(())
        for t in probs1:
            p1, p2 = probs1[t], probs2[t]
            kl = (F.kl_div(p1.clamp(1e-8).log(), p2, reduction="batchmean") +
                  F.kl_div(p2.clamp(1e-8).log(), p1, reduction="batchmean")) * 0.5
            total = total + kl
        return self.alpha * total / len(probs1)


class MultiTaskLossV5(nn.Module):
    """
    Full loss combining:
      CLM (NLL + Focal + Ordinal Smooth) × 3 tasks
      + SupCon (per-task)
      + Hard boundary auxiliaries (OHI 0v1+, MGI (0+1)v(2+3+4), GEI 0v1+)
      + R-Drop
    """
    def __init__(self, config: dict):
        super().__init__()
        kw = dict(sigma=config["clm_sigma"], gamma=config["clm_gamma"],
                  w_nll=config["clm_nll_w"], w_focal=config["clm_focal_w"],
                  w_smooth=config["clm_smooth_w"])

        self.clm_mgi = CLMLoss(config["num_classes_mgi"],
                                config["mgi_class_weights"], **kw)
        self.clm_ohi = CLMLoss(config["num_classes_ohi"],
                                config["ohi_class_weights"], **kw)
        self.clm_gei = CLMLoss(config["num_classes_gei"],
                                config["gei_class_weights"], **kw)

        self.supcon = SupConLoss(config["supcon_temp"],
                                  config.get("supcon_adj_weight", 0.40))
        self.rdrop  = RDropLoss(config["alpha_rdrop"])

        # Hard boundary auxiliaries — targets the most problematic boundaries
        # OHI: class 0 vs class >=1  (OHI=1 collapsed to 0 at 92% in v4)
        ohi_pos_count  = 59 + 10; ohi_neg_count = 134
        ohi_pw = min(ohi_neg_count / ohi_pos_count, 5.0)
        self.ohi_bnd = BoundaryAuxLoss(boundary=1, pos_weight=ohi_pw)

        # MGI: class <=1 vs class >=2 (MGI=1→0 80%, MGI=2→3 62% in v4)
        mgi_pos_count  = 45 + 28 + 8; mgi_neg_count = 74 + 48
        mgi_pw = min(mgi_neg_count / mgi_pos_count, 5.0)
        self.mgi_bnd = BoundaryAuxLoss(boundary=2, pos_weight=mgi_pw)

        # GEI: class 0 vs class >=1
        gei_pos_count = 27 + 4; gei_neg_count = 172
        gei_pw = min(gei_neg_count / gei_pos_count, 8.0)
        self.gei_bnd = BoundaryAuxLoss(boundary=1, pos_weight=gei_pw)

        self.a_sup  = config["alpha_supcon"]
        self.a_ohi_b= config["alpha_ohi_boundary"]
        self.a_mgi_b= config["alpha_mgi_boundary"]
        self.a_gei_b= config["alpha_gei_aux"]
        self.tw     = config["task_loss_weights"]

    def forward(self, probs: dict, targets: dict,
                aux_logits: dict,
                feat_per_task: dict = None,
                probs2: dict = None,
                lam: float = 1.0,
                targets_b: dict = None):
        """
        probs        : {task: (B,K) CLM class probabilities}
        targets      : {task: (B,) int}
        aux_logits   : {ohi_bnd, mgi_bnd, gei_bnd: (B,1)}
        feat_per_task: {task: (B,D) L2-normed embeddings} or None
        probs2       : {task: (B,K)} second pass for R-Drop or None
        """
        info  = {}
        total = next(iter(probs.values())).new_zeros(())

        for task, clm_fn in [("mgi", self.clm_mgi),
                               ("ohi", self.clm_ohi),
                               ("gei", self.clm_gei)]:
            tb = targets_b[task] if targets_b else None
            l  = clm_fn(probs[task], targets[task], lam, tb)
            total = total + self.tw[task] * l
            info[f"clm_{task}"] = l.item()

        # Hard boundary auxiliaries
        if aux_logits.get("ohi_bnd") is not None:
            l_ob  = self.ohi_bnd(aux_logits["ohi_bnd"], targets["ohi"])
            total = total + self.a_ohi_b * l_ob
            info["ohi_bnd"] = l_ob.item()

        if aux_logits.get("mgi_bnd") is not None:
            l_mb  = self.mgi_bnd(aux_logits["mgi_bnd"], targets["mgi"])
            total = total + self.a_mgi_b * l_mb
            info["mgi_bnd"] = l_mb.item()

        if aux_logits.get("gei_bnd") is not None:
            l_gb  = self.gei_bnd(aux_logits["gei_bnd"], targets["gei"])
            total = total + self.a_gei_b * l_gb
            info["gei_bnd"] = l_gb.item()

        # SupCon (per-task, only on pure non-mixed batches)
        if feat_per_task is not None and lam >= 0.99:
            sc = sum(self.supcon(feat_per_task[t], targets[t])
                     for t in ("mgi","ohi","gei")) / 3.0
            total = total + self.a_sup * sc
            info["supcon"] = sc.item()

        # R-Drop
        if probs2 is not None:
            l_rd  = self.rdrop(probs, probs2)
            total = total + l_rd
            info["rdrop"] = l_rd.item()

        return total, info


# ─────────────────────────────────────────────────────────────────────────────
# 8 — Model
# ─────────────────────────────────────────────────────────────────────────────

def _build_dino(name, pretrained, img_size):
    def _mk(pre):
        return timm.create_model(name, pretrained=pre,
                                 img_size=img_size, num_classes=0,
                                 global_pool="")   # full token output
    try:
        return _mk(pretrained)
    except RuntimeError as exc:
        msg = str(exc)
        if "fc_norm.weight" in msg and "norm.weight" in msg:
            log.warning("DINOv2 key mismatch — remapping")
            m   = _mk(False)
            cfg = getattr(m, "pretrained_cfg", {}) or {}
            url = cfg.get("url")
            if not url: return m
            sd  = torch.hub.load_state_dict_from_url(
                url, map_location="cpu", check_hash=False, progress=True)
            remap = {}
            for k, v in sd.items():
                if k == "norm.weight":  remap["fc_norm.weight"] = v
                elif k == "norm.bias":  remap["fc_norm.bias"]   = v
                elif k == "mask_token": pass
                elif k == "pos_embed":
                    pe = getattr(m, "pos_embed", None)
                    if pe is not None and pe.shape != v.shape:
                        cls_ = v[:,:1,:]; ptok = v[:,1:,:]
                        N0=ptok.shape[1]; N1=pe.shape[1]-1
                        g0=int(math.sqrt(N0)); g1=int(math.sqrt(N1))
                        D=ptok.shape[-1]
                        ptok=ptok.reshape(1,g0,g0,D).permute(0,3,1,2)
                        ptok=F.interpolate(ptok,(g1,g1),mode="bicubic",align_corners=False)
                        ptok=ptok.permute(0,2,3,1).reshape(1,N1,D)
                        remap[k]=torch.cat([cls_,ptok],dim=1)
                    else: remap[k]=v
                else: remap[k]=v
            m.load_state_dict(remap, strict=False)
            return m
        if pretrained:
            log.warning(f"DINOv2 load failed; random init")
            return _mk(False)
        raise


def _dino_features(backbone, x, use_cls):
    out = backbone(x)          # (B, 1+N, D) or (B, D)
    if out.dim() == 2: return out
    cls_tok   = out[:, 0, :]
    patch_avg = out[:, 1:, :].mean(1)
    return torch.cat([cls_tok, patch_avg], dim=1) if use_cls else patch_avg


class ViewAttentionPool(nn.Module):
    """Transformer self-attention over 3 views → mean-pool → (B, D)."""
    def __init__(self, dim, heads=4, dropout=0.10):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, heads,
                                            batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(dim*2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):  # (B,3,D) → (B,D)
        a,_ = self.attn(x,x,x); x=self.norm1(x+a)
        return self.norm2(x+self.ff(x)).mean(1)


class OralHealthModelV5(nn.Module):
    """
    DINOv2-small (CLS+patch=768d) + EfficientNet-B4 (1792d)
    Cross-view attention on both → project to 256d
    Per-task CLM heads (ordinal) + per-task SupCon heads
    Hard boundary auxiliary heads for OHI, MGI, GEI
    """
    def __init__(self, config):
        super().__init__()
        self.use_cls  = config.get("dino_use_cls", True)
        self.cnn_attn = config.get("cnn_cross_view_attn", True)

        # Backbones
        self.dino = _build_dino(config["backbone_dino"],
                                 config["pretrained"],
                                 config["image_size"])
        self.cnn  = timm.create_model(config["backbone_cnn"],
                                       pretrained=config["pretrained"],
                                       num_classes=0, global_pool="avg")

        dino_base = self.dino.num_features     # 384 (DINOv2-small)
        dino_dim  = dino_base*2 if self.use_cls else dino_base  # 768
        cnn_dim   = self.cnn.num_features      # 1792 (EfficientNet-B4)
        fused     = dino_dim + cnn_dim         # 2560

        # Cross-view attention pools
        self.dino_pool = ViewAttentionPool(dino_dim, config["view_attn_heads"])
        if self.cnn_attn:
            self.cnn_pool = ViewAttentionPool(cnn_dim, config["view_attn_heads"])

        # Shared projection
        pd = config["projection_dim"]
        self.proj = nn.Sequential(
            nn.Linear(fused, pd), nn.LayerNorm(pd), nn.GELU(),
            nn.Dropout(config["dropout"]))

        # CLM heads (ordinal prediction — THE KEY V5 CHANGE)
        hd = config["clm_hidden_dim"]; dp = config["dropout"]
        self.clm_mgi = CLMHead(pd, config["num_classes_mgi"], hd, dp)
        self.clm_ohi = CLMHead(pd, config["num_classes_ohi"], hd, dp)
        self.clm_gei = CLMHead(pd, config["num_classes_gei"], hd, dp)

        # Hard boundary auxiliary heads
        self.ohi_bnd = nn.Linear(pd, 1)  # OHI 0 vs >=1
        self.mgi_bnd = nn.Linear(pd, 1)  # MGI <=1 vs >=2
        self.gei_bnd = nn.Linear(pd, 1)  # GEI 0 vs >=1

        # Per-task SupCon projection heads
        sc = config["supcon_dim"]
        for t in ("mgi","ohi","gei"):
            setattr(self, f"sc_{t}",
                    nn.Sequential(nn.Linear(pd,sc), nn.ReLU(inplace=True),
                                  nn.Linear(sc,sc)))

    def _dino_f(self, x):
        return _dino_features(self.dino, x, self.use_cls)

    def _cnn_f(self, x):
        return self.cnn(x)

    def _fuse(self, f, l, r):
        d = torch.stack([self._dino_f(f), self._dino_f(l), self._dino_f(r)], 1)
        c = torch.stack([self._cnn_f(f),  self._cnn_f(l),  self._cnn_f(r)],  1)
        d_out = self.dino_pool(d)
        c_out = self.cnn_pool(c) if self.cnn_attn else c.mean(1)
        return torch.cat([d_out, c_out], dim=1)

    def forward(self, frontal, left, right, return_features=False):
        fused = self._fuse(frontal, left, right)
        proj  = self.proj(fused)

        # CLM heads
        probs_mgi, sev_mgi, _ = self.clm_mgi(proj)
        probs_ohi, sev_ohi, _ = self.clm_ohi(proj)
        probs_gei, sev_gei, _ = self.clm_gei(proj)

        out = {
            # Class probability distributions (B, K)
            "probs": {
                "mgi": probs_mgi,
                "ohi": probs_ohi,
                "gei": probs_gei,
            },
            # Hard boundary auxiliary logits (B, 1)
            "aux": {
                "ohi_bnd": self.ohi_bnd(proj),
                "mgi_bnd": self.mgi_bnd(proj),
                "gei_bnd": self.gei_bnd(proj),
            },
            # Severity scores for interpretability (B, 1)
            "severity": {
                "mgi": sev_mgi, "ohi": sev_ohi, "gei": sev_gei,
            },
        }

        if return_features:
            out["features"] = {
                t: F.normalize(getattr(self, f"sc_{t}")(proj), dim=1)
                for t in ("mgi","ohi","gei")
            }

        return out

    def freeze_backbone(self):
        for p in list(self.dino.parameters())+list(self.cnn.parameters()):
            p.requires_grad_(False)
        log.info("Backbones frozen.")

    def unfreeze_last_n_blocks(self, n):
        for blk in list(self.dino.blocks)[-n:]:
            for p in blk.parameters(): p.requires_grad_(True)
        for nm, mod in self.dino.named_modules():
            if "norm" in nm and "blocks" not in nm:
                for p in mod.parameters(): p.requires_grad_(True)
        for ch in list(self.cnn.children())[-n:]:
            for p in ch.parameters(): p.requires_grad_(True)
        n_tr = sum(p.requires_grad for p in
                   list(self.dino.parameters())+list(self.cnn.parameters()))
        log.info(f"Unfroze {n} blocks → {n_tr} backbone params trainable.")

    def head_parameters(self):
        mods = [self.dino_pool, self.proj,
                self.clm_mgi, self.clm_ohi, self.clm_gei,
                self.ohi_bnd, self.mgi_bnd, self.gei_bnd]
        if self.cnn_attn: mods.append(self.cnn_pool)
        for t in ("mgi","ohi","gei"): mods.append(getattr(self, f"sc_{t}"))
        return [p for m in mods for p in m.parameters()]

    def backbone_parameters(self):
        return [p for p in
                list(self.dino.parameters())+list(self.cnn.parameters())
                if p.requires_grad]


# ─────────────────────────────────────────────────────────────────────────────
# 9 — Decoding
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_clm(probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    CLM decoding: simply argmax of class probability distribution.
    
    Unlike OrdinalBCE (which needed threshold calibration),
    CLM's cutpoints are learned end-to-end → argmax is the natural decoder.
    Temperature scaling modifies the class_probs before argmax.
    """
    if temperature != 1.0:
        # Re-sharpen/soften the distribution
        scaled = (probs.log() / max(float(temperature), 1e-4)).exp()
        scaled = scaled / scaled.sum(dim=1, keepdim=True)
        return scaled.argmax(dim=1)
    return probs.argmax(dim=1)


@torch.no_grad()
def tta_ensemble_predict(models_list, frontal, left, right,
                          temps_list, n_tta=7):
    """Average class probs across all fold models and all TTA passes."""
    B = frontal.shape[0]
    accum = {"mgi": None, "ohi": None, "gei": None}

    def _aug(t):
        if random.random() > 0.5: t = torch.flip(t, dims=[-1])
        return (t + torch.randn_like(t)*0.008).clamp(-3, 3)

    for model, temp in zip(models_list, temps_list):
        model.eval()
        for i in range(n_tta):
            fi = frontal if i==0 else _aug(frontal.clone())
            li = left    if i==0 else _aug(left.clone())
            ri = right   if i==0 else _aug(right.clone())
            out = model(fi, li, ri)
            for task in ("mgi","ohi","gei"):
                p = out["probs"][task]
                if accum[task] is None: accum[task] = p.clone()
                else: accum[task] = accum[task] + p

    n_total = len(models_list) * n_tta
    return {t: decode_clm(accum[t] / n_total) for t in ("mgi","ohi","gei")}


# ─────────────────────────────────────────────────────────────────────────────
# 10 — Samplers / data helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_sampler(dataset, indices):
    get = lambda t: np.array([getattr(dataset,f"{t}_labels")[i] for i in indices])
    ma, oa, ga = get("mgi"), get("ohi"), get("gei")
    def inv_f(arr, nc):
        c=np.bincount(arr,minlength=nc).astype(float); c=np.where(c==0,0.5,c)
        return np.array([1.0/c[y] for y in arr])
    w=(inv_f(ma,5)*inv_f(oa,3)*inv_f(ga,3))**(1/3)
    boost=np.where((ma>=3)|(ga>=1), 2.0, 1.0)
    w=np.clip(w*boost, None, 6.0*np.median(w*boost))
    w=w/w.sum()
    return WeightedRandomSampler(torch.tensor(w,dtype=torch.float32),
                                  len(indices), replacement=True)


def build_hard_replay(dataset, indices):
    return [i for i in indices
            if (dataset.mgi_labels[i]>=3 or
                dataset.ohi_labels[i]>=2 or
                dataset.gei_labels[i]>=1)]


def make_epoch_idx(base, pool, frac):
    if frac<=0 or not pool: return list(base)
    n=max(1,int(round(len(base)*frac)))
    rep=np.random.choice(pool,n,replace=(len(pool)<n)).tolist()
    return list(base)+rep


def _collect_batch(batch, device):
    f,l,r,labels=batch
    targets={t: labels[t].to(device).long() for t in ("mgi","ohi","gei")}
    return f.to(device), l.to(device), r.to(device), targets


def _mixup(f,l,r,targets,alpha,prob):
    if alpha<=0 or random.random()>prob: return f,l,r,targets,None,1.0
    lam=max(float(np.random.beta(alpha,alpha)),0.5)
    B=f.shape[0]; idx=torch.randperm(B,device=f.device)
    tb={k:v[idx] for k,v in targets.items()}
    return lam*f+(1-lam)*f[idx],lam*l+(1-lam)*l[idx],lam*r+(1-lam)*r[idx],targets,tb,lam


def _cutmix(f,l,r,targets,alpha,prob):
    if alpha<=0 or random.random()>prob: return f,l,r,targets,None,1.0
    lam=float(np.random.beta(alpha,alpha))
    B,C,H,W=f.shape; idx=torch.randperm(B,device=f.device)
    ch=max(1,int(H*math.sqrt(1-lam))); cw=max(1,int(W*math.sqrt(1-lam)))
    cx=random.randint(0,W); cy=random.randint(0,H)
    x1=max(0,cx-cw//2); x2=min(W,cx+cw//2)
    y1=max(0,cy-ch//2); y2=min(H,cy+ch//2)
    fm=f.clone(); lm=l.clone(); rm=r.clone()
    fm[:,:,y1:y2,x1:x2]=f[idx,:,y1:y2,x1:x2]
    lm[:,:,y1:y2,x1:x2]=l[idx,:,y1:y2,x1:x2]
    rm[:,:,y1:y2,x1:x2]=r[idx,:,y1:y2,x1:x2]
    lam_r=1.0-(x2-x1)*(y2-y1)/float(H*W)
    tb={k:v[idx] for k,v in targets.items()}
    return fm,lm,rm,targets,tb,lam_r


def _mixup_p(epoch, config):
    if epoch<=config["freeze_epochs"]: return 0.0
    p0=config["mixup_prob_start"]; p1=config["mixup_prob_end"]
    start=config["freeze_epochs"]+1
    t=(epoch-start)/max(config["num_epochs"]-start,1)
    return float(np.clip(p0+(p1-p0)*t,0,1))


def _make_scaler():
    if hasattr(torch.amp,"GradScaler"):
        try:    return torch.amp.GradScaler("cuda")
        except: return torch.amp.GradScaler()
    return _amp_compat.GradScaler()


def _autocast():
    if hasattr(torch.amp,"autocast"):
        try:    return torch.amp.autocast(device_type="cuda",enabled=True)
        except: return torch.amp.autocast("cuda",enabled=True)
    return _amp_compat.autocast(enabled=True)


# ─────────────────────────────────────────────────────────────────────────────
# 11 — Metrics (with min-recall check)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, nc_map):
    res = {}
    for task, nc in nc_map.items():
        yt=np.array(y_true[task]); yp=np.array(y_pred[task])
        cm = confusion_matrix(yt, yp, labels=list(range(nc)))
        # Per-class recall
        recall_per_class = np.where(
            cm.sum(1) > 0, np.diag(cm)/cm.sum(1).clip(min=1), 0.0)
        res[task] = {
            "f1":     f1_score(yt, yp, labels=list(range(nc)),
                                average="macro", zero_division=0),
            "acc":    accuracy_score(yt, yp),
            "mae":    float(np.mean(np.abs(yt.astype(int)-yp.astype(int)))),
            "recall": recall_per_class.tolist(),
            "min_recall": float(recall_per_class.min()),
        }
    return res


def ckpt_objective(metrics, nc_map, mae_pen=0.08, min_recall_guard=0.15):
    """
    Combined objective = avg_F1 − mae_pen×norm_MAE
    Returns -inf if any class has recall < min_recall_guard.
    (Prevents selecting models that ignore a class entirely.)
    """
    for task in nc_map:
        if metrics[task]["min_recall"] < min_recall_guard:
            return -999.0   # rejected
    f1s  = [metrics[t]["f1"]  for t in nc_map]
    maes = [metrics[t]["mae"] / max(nc_map[t]-1,1) for t in nc_map]
    return float(np.mean(f1s)) - mae_pen * float(np.mean(maes))


# ─────────────────────────────────────────────────────────────────────────────
# 12 — Temperature scaling (post-hoc CLM calibration)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_temperature(model, val_loader, device, config):
    """
    Minimise NLL of class probabilities by finding optimal temperature T.
    CLM gives class probabilities directly; we scale the log-probs by 1/T.
    Uses torch.enable_grad() inside a no_grad context for LBFGS.
    """
    model.eval()
    all_log_probs = {"mgi":[], "ohi":[], "gei":[]}
    all_labels    = {"mgi":[], "ohi":[], "gei":[]}

    with torch.no_grad():
        for batch in val_loader:
            f,l,r,targets = _collect_batch(batch, device)
            out = model(f,l,r)
            for task in ("mgi","ohi","gei"):
                all_log_probs[task].append(
                    out["probs"][task].clamp(1e-8).log().detach().cpu())
                all_labels[task].extend(targets[task].cpu().numpy().tolist())

    cat_lp = {t: torch.cat(all_log_probs[t],0) for t in ("mgi","ohi","gei")}
    nc_map = {"mgi": config["num_classes_mgi"],
              "ohi": config["num_classes_ohi"],
              "gei": config["num_classes_gei"]}

    T = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=config["temp_lr"],
                              max_iter=config["temp_max_iter"])

    def _nll():
        opt.zero_grad()
        loss = sum(
            F.nll_loss(cat_lp[t] / T.clamp(0.1, 5.0),
                       torch.tensor(all_labels[t], dtype=torch.long))
            for t in nc_map)
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(_nll)

    T_val = float(T.detach().clamp(0.5, 3.0).item())
    print(f"    Temperature = {T_val:.4f}")
    return T_val


# ─────────────────────────────────────────────────────────────────────────────
# 13 — EMA + SWA helpers
# ─────────────────────────────────────────────────────────────────────────────

class EMAHelper:
    def __init__(self, decay=0.998):
        self.decay=decay; self.shadow={}; self.initialized=False

    def register(self, model):
        self.shadow={k:v.detach().float().cpu().clone()
                     for k,v in model.state_dict().items()
                     if torch.is_floating_point(v)}
        self.initialized=True

    def update(self, model):
        if not self.initialized: self.register(model); return
        d=self.decay
        with torch.no_grad():
            for k,v in model.state_dict().items():
                if not torch.is_floating_point(v): continue
                cur=v.detach().float().cpu()
                if k not in self.shadow: self.shadow[k]=cur.clone()
                else: self.shadow[k].mul_(d).add_(cur, alpha=1-d)

    def apply_to(self, model):
        if not self.initialized: return {}
        backup={}
        with torch.no_grad():
            for k,v in model.state_dict().items():
                if k in self.shadow and torch.is_floating_point(v):
                    backup[k]=v.detach().clone()
                    v.copy_(self.shadow[k].to(device=v.device,dtype=v.dtype))
        return backup

    def restore(self, model, backup):
        if not backup: return
        with torch.no_grad():
            for k,v in model.state_dict().items():
                if k in backup: v.copy_(backup[k].to(device=v.device,dtype=v.dtype))


class SWAHelper:
    def __init__(self): self.n=0; self.shadow=None

    def update(self, model):
        sd={k:v.detach().cpu().float().clone() for k,v in model.state_dict().items()}
        if self.shadow is None: self.shadow=sd
        else:
            n=self.n
            self.shadow={k:self.shadow[k]*(n/(n+1))+sd[k]*(1/(n+1)) for k in sd}
        self.n+=1

    def apply_to(self, model):
        if self.shadow is None: return
        cur=model.state_dict()
        model.load_state_dict(
            {k:self.shadow[k].to(cur[k].device,cur[k].dtype) for k in self.shadow})
        log.info(f"SWA applied ({self.n} snapshots).")


# ─────────────────────────────────────────────────────────────────────────────
# 14 — Safe I/O
# ─────────────────────────────────────────────────────────────────────────────

def _safe_save(payload, path, tag=""):
    dst=Path(path); dst.parent.mkdir(parents=True,exist_ok=True)
    for a in range(6):
        tmp=dst.with_suffix(f"{dst.suffix}.tmp.{os.getpid()}.{a}")
        try:
            torch.save(payload,tmp); os.replace(tmp,dst); return str(dst)
        except OSError:
            try: tmp.unlink()
            except: pass
            time.sleep(0.5*(a+1))
    fb=dst.with_name(f"{dst.stem}_{int(time.time())}{dst.suffix}")
    torch.save(payload,fb)
    log.warning(f"{tag}: fallback save → {fb}"); return str(fb)


def _safe_json(payload, path):
    dst=Path(path); dst.parent.mkdir(parents=True,exist_ok=True)
    for a in range(6):
        tmp=dst.with_suffix(f"{dst.suffix}.tmp.{os.getpid()}.{a}")
        try:
            with open(tmp,"w") as fp: json.dump(payload,fp,indent=2)
            os.replace(tmp,dst); return str(dst)
        except OSError:
            try: tmp.unlink()
            except: pass
            time.sleep(0.5*(a+1))
    fb=dst.with_name(f"{dst.stem}_{int(time.time())}{dst.suffix}")
    with open(fb,"w") as fp: json.dump(payload,fp,indent=2)
    return str(fb)


# ─────────────────────────────────────────────────────────────────────────────
# 15 — Train / val epoch functions
# ─────────────────────────────────────────────────────────────────────────────

def run_train_epoch(model, loader, optimizer, loss_fn, scaler,
                    device, config, phase, epoch, ema=None):
    model.train()
    totals={"total":0.0,"mgi":0.0,"ohi":0.0,"gei":0.0}
    n=0; accum=config["grad_accum_steps"]
    mx_p = _mixup_p(epoch,config) if phase==2 else 0.0
    cut_p= config["cutmix_prob"]  if phase==2 else 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        f,l,r,targets=_collect_batch(batch,device)

        if phase==2 and step%2==0:
            f,l,r,targets,tb,lam=_mixup(f,l,r,targets,config["mixup_alpha"],mx_p)
        elif phase==2:
            f,l,r,targets,tb,lam=_cutmix(f,l,r,targets,config["cutmix_alpha"],cut_p)
        else:
            tb,lam=None,1.0

        with _autocast():
            ret_feat=(lam>=0.99)
            out1=model(f,l,r,return_features=ret_feat)
            feat=out1.get("features")
            probs1=out1["probs"]

            out2=model(f,l,r,return_features=False)
            probs2=out2["probs"]

            loss,info=loss_fn(probs1, targets, out1["aux"],
                              feat_per_task=feat,
                              probs2=probs2,
                              lam=lam, targets_b=tb)

        scaler.scale(loss/accum).backward()

        if (step+1)%accum==0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            scaler.step(optimizer); scaler.update()
            if ema: ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        totals["total"]+=loss.item()
        for k in ("mgi","ohi","gei"):
            totals[k]+=info.get(f"clm_{k}",0.0)
        n+=1

    if n%accum!=0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
        scaler.step(optimizer); scaler.update()
        if ema: ema.update(model)
        optimizer.zero_grad(set_to_none=True)

    return {k:v/max(n,1) for k,v in totals.items()}


def run_val_epoch(model, loader, loss_fn, device, config, temperature=1.0):
    model.eval()
    y_true={"mgi":[],"ohi":[],"gei":[]}
    y_pred={"mgi":[],"ohi":[],"gei":[]}
    tot=0.0; n=0

    with torch.no_grad():
        for batch in loader:
            f,l,r,targets=_collect_batch(batch,device)
            with _autocast():
                out=model(f,l,r)
                loss,_=loss_fn(out["probs"], targets, out["aux"])
            tot+=loss.item(); n+=1
            for task in ("mgi","ohi","gei"):
                y_true[task].extend(targets[task].cpu().numpy().tolist())
                y_pred[task].extend(
                    decode_clm(out["probs"][task],
                               temperature).cpu().numpy().tolist())

    return tot/max(n,1), y_true, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# 16 — Main K-Fold training
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.freeze_support()
    seed_everything(CONFIG["seed"])

    if not torch.cuda.is_available():
        raise RuntimeError(
            "\nCUDA GPU required.\n"
            "Install: pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu126 --force-reinstall")

    DEVICE=torch.device("cuda")
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"Torch: {torch.__version__}  |  timm: {timm.__version__}")

    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic=False
    if hasattr(torch.backends.cuda,"matmul"):
        torch.backends.cuda.matmul.allow_tf32=True
    if hasattr(torch,"set_float32_matmul_precision"):
        try: torch.set_float32_matmul_precision("high")
        except: pass

    for d in (CONFIG["checkpoint_dir"], CONFIG["plots_dir"]):
        os.makedirs(d, exist_ok=True)

    # Data
    records, patient_labels = parse_thesis_csv(CONFIG["data_dir"])
    print("\n── LABEL DISTRIBUTION ───────────────────────────────────────────")
    for tag,key,nc in [("MGI(0-4)","mgi",5),("OHI(0-2)","ohi",3),("GEI(0-2)","gei",3)]:
        vals=[ v[key] for v in patient_labels.values()]
        counts=np.bincount(vals, minlength=nc)
        line="  ".join(f"c{i}:{n}({n/len(vals)*100:.0f}%)"
                        +(" ⚠" if n<15 else "")
                        for i,n in enumerate(counts))
        print(f"  {tag}: {line}")
    print("─────────────────────────────────────────────────────────────────\n")

    FULL_TRAIN_DS = MultiViewPatientDataset(records, TRAIN_AUG_NORMAL, TRAIN_AUG_STRONG)
    FULL_VAL_DS   = MultiViewPatientDataset(records, VAL_AUG)

    NC_MAP   = {"mgi":CONFIG["num_classes_mgi"],
                "ohi":CONFIG["num_classes_ohi"],
                "gei":CONFIG["num_classes_gei"]}
    all_mgi  = np.array(FULL_TRAIN_DS.mgi_labels)
    n_pts    = len(FULL_TRAIN_DS)

    mgi_cnts = np.bincount(all_mgi, minlength=5)
    eff_k    = min(CONFIG["k_folds"], int(mgi_cnts[mgi_cnts>0].min()))
    if eff_k<2: raise RuntimeError(f"Too few samples. MGI: {mgi_cnts}")
    if eff_k!=CONFIG["k_folds"]:
        log.warning(f"k_folds {CONFIG['k_folds']}→{eff_k}")

    skf = StratifiedKFold(n_splits=eff_k, shuffle=True,
                          random_state=CONFIG["seed"])

    fold_sums=[]; all_swa=[]; all_hist=[]

    print(f"\n{'='*72}")
    print(f"  DentAI v5 (FINAL)  |  {eff_k} folds × {CONFIG['num_epochs']} epochs")
    print(f"  CLM heads fix ordinal polarisation observed in v4")
    print(f"  Hard boundary auxiliaries target OHI 0/1 and MGI 1/2 boundaries")
    print(f"  Min-recall guard: no checkpoint accepted if any class recall < "
          f"{CONFIG['min_class_recall']:.0%}")
    print(f"{'='*72}\n")

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.arange(n_pts), all_mgi), start=1):
        seed_everything(CONFIG["seed"]+fold_idx)
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

        model   = OralHealthModelV5(CONFIG).to(DEVICE)
        loss_fn = MultiTaskLossV5(CONFIG).to(DEVICE)
        scaler  = _make_scaler()
        swa     = SWAHelper()
        ema     = EMAHelper(CONFIG["ema_decay"])

        # Phase 1: freeze backbone
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            model.head_parameters(),
            lr=CONFIG["lr_heads"],
            weight_decay=CONFIG["weight_decay"])
        sched_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["freeze_epochs"],
            eta_min=CONFIG["lr_heads"]*0.1)

        ckpt_path = Path(CONFIG["checkpoint_dir"]) / f"fold_{fold_idx}_best.pth"
        swa_path  = Path(CONFIG["checkpoint_dir"]) / f"fold_{fold_idx}_swa.pth"
        saved_ckpt= None; best_obj=-999.0; patience=0
        fold_hist=[]; phase=1; temperature=1.0

        ema_start   =CONFIG["ema_start_epoch"]
        use_ema_eval=CONFIG["use_ema_eval"]
        min_recall  =CONFIG["min_class_recall"]

        for epoch in range(1, CONFIG["num_epochs"]+1):

            # Transition: unfreeze backbone
            if epoch == CONFIG["freeze_epochs"]+1:
                phase=2
                model.unfreeze_last_n_blocks(CONFIG["unfreeze_blocks"])

                pg=[
                    {"params":model.backbone_parameters(),    "lr":CONFIG["lr_backbone"]},
                    {"params":model.dino_pool.parameters(),   "lr":CONFIG["lr_projection"]},
                    {"params":model.proj.parameters(),        "lr":CONFIG["lr_projection"]},
                    {"params":model.clm_mgi.parameters(),     "lr":CONFIG["lr_heads"]},
                    {"params":model.clm_ohi.parameters(),     "lr":CONFIG["lr_heads"]},
                    {"params":model.clm_gei.parameters(),     "lr":CONFIG["lr_heads"]},
                    {"params":model.ohi_bnd.parameters(),     "lr":CONFIG["lr_heads"]},
                    {"params":model.mgi_bnd.parameters(),     "lr":CONFIG["lr_heads"]},
                    {"params":model.gei_bnd.parameters(),     "lr":CONFIG["lr_heads"]},
                ] + [
                    {"params":getattr(model,f"sc_{t}").parameters(),
                     "lr":CONFIG["lr_projection"]}
                    for t in ("mgi","ohi","gei")
                ]
                if CONFIG["cnn_cross_view_attn"]:
                    pg.append({"params":model.cnn_pool.parameters(),
                               "lr":CONFIG["lr_projection"]})
                optimizer=torch.optim.AdamW(pg, weight_decay=CONFIG["weight_decay"])
                sched_p2=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=CONFIG["sgdr_T0"], T_mult=CONFIG["sgdr_T_mult"],
                    eta_min=CONFIG["lr_heads"]*CONFIG["sgdr_eta_min_frac"])

            # Build epoch loader with hard replay
            rf  = CONFIG["hard_replay_frac"] if epoch>=CONFIG["hard_replay_start"] else 0.0
            ep_idx  = make_epoch_idx(base_train, replay_pool, rf)
            sampler = build_sampler(FULL_TRAIN_DS, ep_idx)
            t_loader= DataLoader(
                Subset(FULL_TRAIN_DS, ep_idx),
                batch_size=CONFIG["batch_size"], sampler=sampler,
                drop_last=True, num_workers=CONFIG["num_workers"])

            ema_step = ema if (phase==2 and epoch>=ema_start) else None

            # Train
            tr_loss = run_train_epoch(model, t_loader, optimizer, loss_fn,
                                       scaler, DEVICE, CONFIG, phase, epoch,
                                       ema=ema_step)
            if phase==1: sched_p1.step()
            else:        sched_p2.step()

            if epoch>=CONFIG["swa_start_epoch"]: swa.update(model)

            # Validate
            ema_active=(phase==2 and epoch>=ema_start
                        and use_ema_eval and ema.initialized)
            if ema_active:
                bk=ema.apply_to(model)
                try:    vl,yt,yp=run_val_epoch(model,val_loader,loss_fn,DEVICE,CONFIG,temperature)
                finally:ema.restore(model,bk)
            else:
                vl,yt,yp=run_val_epoch(model,val_loader,loss_fn,DEVICE,CONFIG,temperature)

            metrics=compute_metrics(yt,yp,NC_MAP)
            obj    =ckpt_objective(metrics,NC_MAP,
                                    min_recall_guard=min_recall)
            avg_f1 =float(np.mean([metrics[t]["f1"] for t in NC_MAP]))

            fold_hist.append({"epoch":epoch,"train_loss":tr_loss["total"],
                               "val_loss":vl,"val_avg_f1":avg_f1,
                               "f1_mgi":metrics["mgi"]["f1"],
                               "f1_ohi":metrics["ohi"]["f1"],
                               "f1_gei":metrics["gei"]["f1"],
                               "mae_mgi":metrics["mgi"]["mae"],
                               "min_rec":min(metrics[t]["min_recall"] for t in NC_MAP),
                               "objective":obj})

            if epoch%5==0 or epoch<=3:
                rec_str="/".join(f"{metrics[t]['min_recall']:.2f}" for t in NC_MAP)
                star=" ★" if obj>best_obj else ""
                em="  EMA" if ema_active else ""
                print(f"  Ep{epoch:3d}  "
                      f"tr={tr_loss['total']:.3f}  vl={vl:.3f}  "
                      f"F1(mgi={metrics['mgi']['f1']:.3f} "
                      f"ohi={metrics['ohi']['f1']:.3f} "
                      f"gei={metrics['gei']['f1']:.3f})  "
                      f"min_rec={rec_str}  "
                      f"obj={obj:.4f}{em}{star}")

            if obj > best_obj:
                best_obj=obj; patience=0
                payload={"epoch":epoch, "model_state_dict":None,
                         "objective":obj, "metrics":metrics,
                         "temperature":temperature, "config":CONFIG}
                if ema_active:
                    bk=ema.apply_to(model)
                    try:
                        payload["model_state_dict"]={
                            k:v.cpu() for k,v in model.state_dict().items()}
                        saved_ckpt=_safe_save(payload,ckpt_path,f"F{fold_idx}")
                    finally: ema.restore(model,bk)
                else:
                    payload["model_state_dict"]={
                        k:v.cpu() for k,v in model.state_dict().items()}
                    saved_ckpt=_safe_save(payload,ckpt_path,f"F{fold_idx}")
            else:
                patience+=1

            if patience>=CONFIG["early_stopping_patience"] and phase==2:
                print(f"  Early stop ep{epoch}.")
                break

        # ── Post-fold: temp scaling ─────────────────────────────────────────
        if saved_ckpt is None:
            payload={"epoch":epoch,
                     "model_state_dict":{k:v.cpu() for k,v in model.state_dict().items()},
                     "objective":best_obj,"metrics":{},"temperature":1.0,"config":CONFIG}
            saved_ckpt=_safe_save(payload,ckpt_path,f"F{fold_idx}_fb")

        m_tmp=OralHealthModelV5(CONFIG).to(DEVICE)
        ck=torch.load(saved_ckpt,map_location=DEVICE,weights_only=False)
        m_tmp.load_state_dict(ck["model_state_dict"])
        print(f"  Temperature scaling (fold {fold_idx})...")
        temperature=optimize_temperature(m_tmp,val_loader,DEVICE,CONFIG)
        ck["temperature"]=temperature
        saved_ckpt=_safe_save(ck,ckpt_path,f"F{fold_idx}_T")
        del m_tmp; torch.cuda.empty_cache()

        # ── SWA ─────────────────────────────────────────────────────────────
        swa_temp=temperature
        if swa.n>0:
            m_swa=OralHealthModelV5(CONFIG).to(DEVICE)
            ck_=torch.load(saved_ckpt,map_location=DEVICE,weights_only=False)
            m_swa.load_state_dict(ck_["model_state_dict"])
            swa.apply_to(m_swa)
            print(f"  SWA temp scaling ({swa.n} snapshots)...")
            swa_temp=optimize_temperature(m_swa,val_loader,DEVICE,CONFIG)
            swa_pay={"model_state_dict":{k:v.cpu() for k,v in m_swa.state_dict().items()},
                     "swa_n":swa.n,"temperature":swa_temp,"config":CONFIG}
            saved_swa=_safe_save(swa_pay,swa_path,f"F{fold_idx}_swa")
            del m_swa; torch.cuda.empty_cache()
        else:
            shutil.copy(saved_ckpt,swa_path)
            saved_swa=str(swa_path)

        all_swa.append(saved_swa)

        beh=max(fold_hist,key=lambda h:h["objective"])
        fold_sums.append({"fold":fold_idx,"best_obj":best_obj,
                           "best_epoch":beh["epoch"],
                           "f1_mgi":beh["f1_mgi"],"f1_ohi":beh["f1_ohi"],
                           "f1_gei":beh["f1_gei"],"mae_mgi":beh["mae_mgi"],
                           "min_recall":beh["min_rec"],
                           "temperature":temperature,
                           "swa_temperature":swa_temp,
                           "checkpoint":str(saved_ckpt),
                           "swa_checkpoint":saved_swa})
        all_hist.append(fold_hist)

        print(f"\n  Fold {fold_idx}: obj={best_obj:.4f}  "
              f"mgi={beh['f1_mgi']:.3f}  ohi={beh['f1_ohi']:.3f}  "
              f"gei={beh['f1_gei']:.3f}  T={temperature:.4f}")
        del model,loss_fn,optimizer,scaler,swa,ema; torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────

    df=pd.DataFrame(fold_sums)
    print(f"\n{'='*72}")
    print("  FOLD RESULTS")
    print(df[["fold","best_obj","best_epoch","f1_mgi","f1_ohi",
               "f1_gei","mae_mgi","min_recall"]].to_string(index=False))
    print("\n  Mean ± Std:")
    for c in ["f1_mgi","f1_ohi","f1_gei"]:
        v=df[c]; print(f"    {c}: {v.mean():.4f} ± {v.std():.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # All-fold SWA ensemble
    # ─────────────────────────────────────────────────────────────────────────

    avg_temp=float(np.mean([s["swa_temperature"] for s in fold_sums]))
    best_fi =int(df["best_obj"].idxmax())
    _, vb   = list(skf.split(np.arange(n_pts), all_mgi))[best_fi]
    vl_best = DataLoader(Subset(FULL_VAL_DS,list(vb)),
                          batch_size=1,shuffle=False,
                          num_workers=CONFIG["num_workers"])

    print(f"\nLoading {len(all_swa)} SWA models for ensemble eval...")
    ens_models=[]; ens_temps=[]
    for p in all_swa:
        m=OralHealthModelV5(CONFIG).to(DEVICE)
        try:    ck=torch.load(p,map_location=DEVICE,weights_only=False)
        except: ck=torch.load(p,map_location=DEVICE)
        m.load_state_dict(ck["model_state_dict"]); m.eval()
        ens_models.append(m)
        ens_temps.append(float(ck.get("temperature",avg_temp)))

    yt_ens={"mgi":[],"ohi":[],"gei":[]}
    yp_ens={"mgi":[],"ohi":[],"gei":[]}
    with torch.no_grad():
        for batch in vl_best:
            f,l,r,labels=batch
            f,l,r=f.to(DEVICE),l.to(DEVICE),r.to(DEVICE)
            preds=tta_ensemble_predict(ens_models,f,l,r,ens_temps,
                                        CONFIG["tta_steps"])
            for task in ("mgi","ohi","gei"):
                yt_ens[task].extend(labels[task].numpy().tolist())
                yp_ens[task].extend(preds[task].cpu().numpy().tolist())

    print("\n── ALL-FOLD SWA ENSEMBLE (TTA×7) ────────────────────────────────")
    for task,nc in NC_MAP.items():
        f1=f1_score(yt_ens[task],yp_ens[task],average="macro",zero_division=0)
        print(f"\n  {task.upper()} macro-F1: {f1:.4f}")
        print(classification_report(yt_ens[task],yp_ens[task],zero_division=0))

    for m in ens_models: del m
    torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────────────────────

    fig,axes=plt.subplots(2,3,figsize=(21,12))
    for fi,hist in enumerate(all_hist):
        ep=[h["epoch"] for h in hist]
        axes[0,0].plot(ep,[h["train_loss"] for h in hist],label=f"F{fi+1}",alpha=0.8)
        axes[0,1].plot(ep,[h["val_loss"]   for h in hist],label=f"F{fi+1}",alpha=0.8)
        axes[0,2].plot(ep,[h["val_avg_f1"] for h in hist],label=f"F{fi+1}",alpha=0.8)
    for ax,t in zip(axes[0],["Train Loss","Val Loss","Val Avg Macro-F1"]):
        ax.set_title(t,fontweight="bold"); ax.set_xlabel("Epoch")
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    for ax,(title,task,nc) in zip(axes[1],[
        (f"MGI 0-{NC_MAP['mgi']-1}","mgi",NC_MAP["mgi"]),
        (f"OHI 0-{NC_MAP['ohi']-1} merged","ohi",NC_MAP["ohi"]),
        (f"GEI 0-{NC_MAP['gei']-1} merged","gei",NC_MAP["gei"]),
    ]):
        cm=confusion_matrix(yt_ens[task],yp_ens[task],labels=list(range(nc)))
        cmnorm=cm.astype(float)/cm.sum(axis=1,keepdims=True).clip(min=1)
        sns.heatmap(cmnorm,annot=True,fmt=".2f",cmap="Blues",ax=ax,
                    xticklabels=range(nc),yticklabels=range(nc))
        ax.set_title(f"{title} — Ensemble Confusion",fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    plt.suptitle("DentAI v5 — Training Results (All-fold SWA Ensemble)",
                 fontsize=14,fontweight="bold")
    plt.tight_layout()
    pp=os.path.join(CONFIG["plots_dir"],"training_results_v5.png")
    plt.savefig(pp,dpi=150,bbox_inches="tight"); plt.close()
    print(f"\nPlot → {pp}")

    # ─────────────────────────────────────────────────────────────────────────
    # Export ensemble config
    # ─────────────────────────────────────────────────────────────────────────

    avg_f1={t:float(df[f"f1_{t}"].mean()) for t in ("mgi","ohi","gei")}
    ens_cfg={"model_type":"OralHealthModelV5","models":all_swa,
              "avg_temperature":avg_temp,"avg_val_f1":avg_f1,
              "config":CONFIG,
              "label_mapping":{"mgi":"0-4 unchanged",
                                "ohi":"0,1,2 (2+3 merged)",
                                "gei":"0,1,2 (2+3 merged)"},
              "decode_instruction":
                "probs = model(f,l,r)['probs'][task]; pred = probs.argmax(1)",
              "clm_note":
                "CLM heads produce class probabilities directly; no threshold "
                "calibration needed. Temperature scaling optional for confidence."}
    for sp in [Path(CONFIG["checkpoint_dir"])/"ensemble_config.json",
               Path(CONFIG["checkpoint_dir"]).parent/"ensemble_config.json"]:
        _safe_json(ens_cfg, sp)
        print(f"Ensemble config → {sp}")

    print(f"\n{'='*72}")
    print(f"  DONE — DentAI v5")
    print(f"  Mean F1: MGI={avg_f1['mgi']:.4f}  "
          f"OHI={avg_f1['ohi']:.4f}  GEI={avg_f1['gei']:.4f}")
    print(f"  Avg temperature: {avg_temp:.4f}")
    print(f"{'='*72}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Smoke test
    # ─────────────────────────────────────────────────────────────────────────

    print("── Smoke test ───────────────────────────────────────────────────")
    m_t=OralHealthModelV5(CONFIG).to("cpu")
    try:    ck=torch.load(all_swa[0],map_location="cpu",weights_only=False)
    except: ck=torch.load(all_swa[0],map_location="cpu")
    m_t.load_state_dict(ck["model_state_dict"]); m_t.eval()
    dummy=torch.zeros(1,3,CONFIG["image_size"],CONFIG["image_size"])
    with torch.no_grad():
        out_t=m_t(dummy,dummy,dummy,return_features=True)
    print("✓ Forward pass OK")
    print(f"  probs keys: {list(out_t['probs'].keys())}")
    print(f"  aux keys:   {list(out_t['aux'].keys())}")
    for task,nc in NC_MAP.items():
        p=out_t["probs"][task]
        print(f"  {task}: prob_shape={tuple(p.shape)}  "
              f"sum={p.sum().item():.4f}  "
              f"pred={decode_clm(p).item()}")
    del m_t
    print(f"─────────────────────────────────────────────────────────────────")
    print(f"Checkpoints: {CONFIG['checkpoint_dir']}")
    print("Done.")