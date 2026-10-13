"""Plaque Index (0-5) from disclosing-agent photographs.

Disclosing solution (erythrosine) stains plaque pink/magenta. This module
measures how much of the visible tooth surface carries that stain and maps
the coverage to a Turesky-modified Quigley-Hein style score (0-5).

Pipeline per view:
  1. Mouth ROI     - convex region enclosed by pink oral tissue; drops skin
                     and most of the cheek retractor.
  2. Tooth seeds   - bright, only mildly pink blobs inside the ROI (Otsu on
                     L*), excluding blobs on the ROI rim and long thin strips
                     (retractor edges, cheek folds).
  3. Stain         - magenta pixels on the tooth or in a thin ring around it,
                     judged relative to this patient's own enamel tint and
                     brightness so colour casts and dyed gingiva (darker) are
                     not counted as plaque.
  4. Coverage      - stain / (stain + clean enamel); views are weighted by
                     detected tooth area.

Use only with photographs taken after applying a disclosing agent. Normal
photographs must go through the MGI/OHI/GEI model instead.

Calibration note (2026-10-13): COVERAGE_THRESHOLDS were set against three
clinician-scored patients (PI 2, 3, 4). Re-fit them with
`calibrate_thresholds()` once more scored cases are available.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

WORK_SIZE = 900  # longest side used for analysis

# Coverage ratio cut-offs; PI = number of thresholds exceeded (0-5).
COVERAGE_THRESHOLDS = [0.02, 0.05, 0.10, 0.14, 0.25]

DEFAULT_PARAMS = {
    "pink_a": 14,       # a* of oral tissue used to find the mouth
    "edge": 0.05,       # ROI rim width (fraction of image size)
    "inner_frac": 0.7,  # tooth blobs must lie this much inside the rim
    "max_elong": 4.0,   # reject blobs longer than this x their width
    "l_floor": 0.45,    # minimum tooth brightness vs. brightest pixel
    "br_a": 24,         # tooth seeds may be at most this pink (a*)
    "min_area": 0.0006, # minimum tooth blob area (fraction of image)
    "ring": 0.008,      # ring around teeth searched for stain
    "st_a": 18,         # absolute minimum a* for stain
    "st_delta": 14,     # stain must be this much pinker than the enamel
    "st_l": 0.60,       # stain must be at least this bright vs. enamel
}

PI_LABELS = {
    0: "No plaque",
    1: "Separate flecks of plaque at the cervical margin",
    2: "Thin continuous band of plaque at the cervical margin",
    3: "Band of plaque wider than 1 mm, less than 1/3 of the crown",
    4: "Plaque covering 1/3 to 2/3 of the crown",
    5: "Plaque covering 2/3 or more of the crown",
}


def _ellipse(r: float) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, thickness=cv2.FILLED)
    return out


def _resize(image_rgb: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    s = WORK_SIZE / max(h, w)
    if s >= 1:
        return image_rgb
    return cv2.resize(image_rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _mouth_roi(a: np.ndarray, b: np.ndarray, p: Dict) -> np.ndarray:
    H, W = a.shape
    pink = ((a > p["pink_a"]) & (a > b + 6)).astype(np.uint8) * 255
    pink = cv2.morphologyEx(pink, cv2.MORPH_CLOSE, _ellipse(max(H, W) * 0.02))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(pink, connectivity=8)
    if n <= 1:
        return np.full((H, W), 255, np.uint8)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lbl == big).astype(np.uint8) * 255,
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    roi = np.zeros((H, W), np.uint8)
    cv2.fillPoly(roi, [hull], 255)
    return roi


def segment_disclosed(image_rgb: np.ndarray, params: Optional[Dict] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (clean_enamel_mask, stain_mask, mouth_roi) at WORK_SIZE scale."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    H, W = L.shape
    D = max(H, W)

    roi = _mouth_roi(a, b, p)
    inside = roi > 0
    inner = cv2.erode(roi, _ellipse(D * p["edge"])) > 0

    # Teeth: the bright class inside the mouth.
    l_in = L[inside].astype(np.uint8)
    otsu, _ = cv2.threshold(l_in.reshape(-1, 1), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    l_thr = max(otsu, p["l_floor"] * np.percentile(l_in, 99.5))
    bright = inside & (L > l_thr) & (a < p["br_a"]) & (b > -4)

    cand = cv2.morphologyEx(bright.astype(np.uint8) * 255, cv2.MORPH_OPEN, _ellipse(2))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    seeds = np.zeros_like(cand)
    min_area = H * W * p["min_area"]
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = lbl == i
        if inner[comp].mean() < p["inner_frac"]:
            continue  # on the rim of the mouth: retractor / lip
        (_, _), (rw, rh), _ = cv2.minAreaRect(cv2.findNonZero(comp.astype(np.uint8)))
        if max(rw, rh) > p["max_elong"] * max(min(rw, rh), 1):
            continue  # long thin strip: cheek fold / retractor edge
        seeds[comp] = 255
    teeth = _fill_holes(cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, _ellipse(3)))

    if not (seeds > 0).any():
        empty = np.zeros_like(seeds)
        return empty, empty, roi

    # Stain relative to this patient's enamel tint and brightness.
    l_teeth = float(np.median(L[seeds > 0]))
    a_teeth = float(np.median(a[seeds > 0]))
    a_thr = max(p["st_a"], a_teeth + p["st_delta"])
    stain = (a > a_thr) & (a > b + 10) & (L > p["st_l"] * l_teeth)
    ring = cv2.dilate(teeth, _ellipse(D * p["ring"]))
    stain_m = (stain.astype(np.uint8) * 255) & ring
    stain_m = cv2.morphologyEx(stain_m, cv2.MORPH_OPEN, _ellipse(1))
    enamel = teeth & ~stain_m
    return enamel, stain_m, roi


def disclosed_coverage(image_rgb: np.ndarray, params: Optional[Dict] = None
                       ) -> Tuple[float, int]:
    """Return (stain coverage ratio, tooth area in px) for one view."""
    enamel, stain, _ = segment_disclosed(_resize(image_rgb), params)
    ea, sa = int((enamel > 0).sum()), int((stain > 0).sum())
    tooth = ea + sa
    return (sa / tooth if tooth else 0.0), tooth


def coverage_to_pi(coverage: float, thresholds: Sequence[float] = COVERAGE_THRESHOLDS) -> int:
    return int(sum(coverage > t for t in thresholds))


def estimate_pi_disclosed(
    frontal_rgb: np.ndarray,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    params: Optional[Dict] = None,
) -> Dict:
    """Estimate PI (0-5) from three disclosing-agent views.

    Returns a dict with pi_score, coverage (weighted stain ratio 0-1),
    per-view coverages, tooth_area_px and a confidence string.
    """
    covs, areas = [], []
    for img in (frontal_rgb, left_rgb, right_rgb):
        c, area = disclosed_coverage(img, params)
        covs.append(c)
        areas.append(area)

    w = np.asarray(areas, dtype=float)
    if w.sum() < 1:
        return {
            "pi_score": None,
            "coverage": None,
            "coverage_f": 0.0, "coverage_l": 0.0, "coverage_r": 0.0,
            "tooth_area_px": 0,
            "confidence": "low (no tooth surface detected)",
        }

    coverage = float(np.dot(w / w.sum(), covs))
    pi_score = coverage_to_pi(coverage)

    # Views that found few teeth or disagree strongly are less trustworthy.
    min_area = min(areas)
    spread = max(covs) - min(covs)
    if min_area > 15000 and spread < 0.15:
        confidence = "high"
    elif min_area > 4000:
        confidence = "medium"
    else:
        confidence = "low (small tooth area detected)"

    return {
        "pi_score": pi_score,
        "coverage": round(coverage, 4),
        "coverage_f": round(covs[0], 4),
        "coverage_l": round(covs[1], 4),
        "coverage_r": round(covs[2], 4),
        "tooth_area_px": int(w.sum()),
        "confidence": confidence,
    }


def calibrate_thresholds(coverages: Sequence[float], true_pi: Sequence[int]) -> list:
    """Fit COVERAGE_THRESHOLDS to clinician-scored cases (needs scipy)."""
    from scipy.optimize import minimize

    cov = np.asarray(coverages, dtype=float)
    y = np.asarray(true_pi, dtype=float)

    def loss(t):
        preds = np.array([coverage_to_pi(c, sorted(t)) for c in cov])
        return float(((preds - y) ** 2).sum())

    res = minimize(loss, COVERAGE_THRESHOLDS, method="Nelder-Mead")
    return sorted(float(v) for v in res.x)


def visualize_disclosed(image_rgb: np.ndarray, params: Optional[Dict] = None) -> np.ndarray:
    """Overlay for debugging: green = clean enamel, magenta = detected stain."""
    img = _resize(image_rgb)
    enamel, stain, roi = segment_disclosed(img, params)
    out = img.copy()
    out[enamel > 0] = (0.5 * out[enamel > 0] + [0, 110, 0]).astype(np.uint8)
    out[stain > 0] = (0.4 * out[stain > 0] + [150, 0, 150]).astype(np.uint8)
    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python disclosed_pi.py <D-frontal> <D-left> <D-right>")
        sys.exit(0)

    def _load(path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = estimate_pi_disclosed(*(_load(p) for p in sys.argv[1:4]))
    for k, v in result.items():
        print(f"  {k:14s}: {v}")
