# PI (Plaque Index 0–5) — Complete Implementation Guide for Copilot
# ===================================================================
# No training data exists for PI, so this is a calibrated rule-based
# computer-vision estimator using the same three intra-oral photos.
#
# Clinical basis (Silness–Löe Plaque Index):
#   0 = No plaque
#   1 = Thin film of plaque at gingival margin (not visible to eye, only on probe)
#   2 = Moderate accumulation — visible soft deposits
#   3 = Abundant soft deposits in the sulcus / on the margin
#   4 = Very heavy accumulation, deposits cover most of tooth surface
#   5 = Generalised / severe — entire crown covered
#
# This guide implements PI estimation in three layers:
#   Layer 1 — Color-based plaque pixel detection (primary)
#   Layer 2 — Tooth surface segmentation (prerequisite)
#   Layer 3 — OHI-proxy calibration (corrects systematic bias)
#
# File layout expected:
#   inference/
#     pi_estimator.py    ← this code goes here
#
# Dependencies already present: cv2, numpy, scipy
# ===================================================================

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Tooth Surface Segmentation
# ─────────────────────────────────────────────────────────────────────────────
#
# Teeth are the brightest, most neutral-coloured region in intra-oral photos.
# Strategy: isolate by HSV range (high V, low S, neutral H), then take the
# largest connected component within the central 60% of the image.
#
# Why the central crop? The palate and cheeks can also be bright/neutral.
# Teeth are always in the centre of intra-oral photos.

def segment_tooth_surface(image_rgb: np.ndarray) -> np.ndarray:
    """
    Returns a binary mask (uint8, 0/255) marking likely tooth pixels.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image

    Returns:
        mask: (H, W) uint8  — 255 = tooth pixel, 0 = background
    """
    H, W = image_rgb.shape[:2]
    hsv  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lab  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    # ── HSV tooth range ────────────────────────────────────────────────────
    # Enamel: high brightness (V > 130), low-moderate saturation (S < 90)
    # Hue is relaxed (0-180) because teeth range from white to ivory
    lower_hsv = np.array([0,   0, 130], dtype=np.uint8)
    upper_hsv = np.array([180, 90, 255], dtype=np.uint8)
    mask_hsv  = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # ── LAB tooth range ────────────────────────────────────────────────────
    # L* > 65 (bright), a* near 0 ± 20, b* 0-35 (slight yellow ok)
    lower_lab = np.array([165,  108, 128], dtype=np.uint8)  # LAB stored 0-255
    upper_lab = np.array([255,  148, 170], dtype=np.uint8)
    mask_lab  = cv2.inRange(lab, lower_lab, upper_lab)

    # Combine
    mask = cv2.bitwise_and(mask_hsv, mask_lab)

    # ── Central crop (middle 60% width, middle 70% height) ────────────────
    region_mask = np.zeros_like(mask)
    y1 = int(H * 0.15); y2 = int(H * 0.85)
    x1 = int(W * 0.10); x2 = int(W * 0.90)
    region_mask[y1:y2, x1:x2] = 255
    mask = cv2.bitwise_and(mask, region_mask)

    # ── Morphological cleanup ──────────────────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # ── Keep only the N largest connected components (the actual teeth) ────
    n_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    if n_labels <= 1:
        return mask  # no components found — return as-is

    # Sort by area, skip background (label 0), keep top 3 largest
    areas    = stats[1:, cv2.CC_STAT_AREA]
    top_idxs = np.argsort(areas)[::-1][:3] + 1

    # Minimum area threshold (avoid tiny noise blobs)
    min_area = (H * W) * 0.005
    clean    = np.zeros_like(mask)
    for idx in top_idxs:
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            clean[labels_map == idx] = 255

    return clean


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Plaque Pixel Detection
# ─────────────────────────────────────────────────────────────────────────────
#
# Plaque appears as:
#   - Yellow/cream deposits  (fresh plaque)
#   - White/opaque films     (biofilm)
#   - Brown/tan stains       (mature/calculus)
#
# Gingiva (red/pink) must be excluded from the tooth mask before this step.
# In LAB space:
#   Plaque: L=60-90, a=-5 to +20, b=+10 to +45
#   Clean enamel: L>70, a near 0, b near 0-20, very low saturation

def detect_plaque_pixels(image_rgb: np.ndarray,
                          tooth_mask: np.ndarray) -> np.ndarray:
    """
    Within the tooth_mask region, identify pixels that look like plaque.

    Returns:
        plaque_mask: (H, W) uint8 — 255 = plaque pixel, 0 = clean
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # ── Gingival margin exclusion (top 20% of tooth mask bounding box) ────
    # Gingiva is immediately above teeth — exclude top strip to avoid false positives
    ys, xs = np.where(tooth_mask > 0)
    if len(ys) == 0:
        return np.zeros_like(tooth_mask)
    y_top    = int(ys.min())
    y_bottom = int(ys.max())
    gingival_strip = max(int((y_bottom - y_top) * 0.15), 5)

    # Exclude the gingival margin strip from the tooth mask for plaque detection
    tooth_body = tooth_mask.copy()
    tooth_body[y_top: y_top + gingival_strip, :] = 0

    # ── LAB plaque range ───────────────────────────────────────────────────
    # L* > 55 (not dark), b* > 12 (yellower than clean enamel)
    # a* slightly positive (plaque is not green-tinted)
    lower_plaque_lab = np.array([140, 120, 137], dtype=np.uint8)  # scaled 0-255
    upper_plaque_lab = np.array([220, 145, 175], dtype=np.uint8)
    plaque_lab       = cv2.inRange(lab, lower_plaque_lab, upper_plaque_lab)

    # ── HSV plaque range ───────────────────────────────────────────────────
    # Yellow-cream: H=20-45, S=30-180, V>100
    lower_plaque_hsv = np.array([15,  25, 100], dtype=np.uint8)
    upper_plaque_hsv = np.array([50, 200, 255], dtype=np.uint8)
    plaque_hsv       = cv2.inRange(hsv, lower_plaque_hsv, upper_plaque_hsv)

    # Combine: pixel must match EITHER LAB OR HSV range
    plaque_combined = cv2.bitwise_or(plaque_lab, plaque_hsv)

    # Restrict to tooth body only (excluding gingival strip)
    plaque_mask = cv2.bitwise_and(plaque_combined, tooth_body)

    # Small morphological cleanup
    kernel2     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    plaque_mask = cv2.morphologyEx(plaque_mask, cv2.MORPH_OPEN,  kernel2)
    plaque_mask = cv2.morphologyEx(plaque_mask, cv2.MORPH_CLOSE, kernel2)

    return plaque_mask


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Coverage Ratio → PI Score
# ─────────────────────────────────────────────────────────────────────────────
#
# Silness–Löe scale based on coverage percentage of visible tooth surface:
#   PI=0: 0%          (no detectable plaque)
#   PI=1: 0–10%       (thin film, often not visible)
#   PI=2: 10–33%      (moderate, visible accumulation)
#   PI=3: 33–55%      (heavy)
#   PI=4: 55–75%      (very heavy)
#   PI=5: 75–100%     (generalised, severe)
#
# Thresholds are approximate clinical guidelines.

COVERAGE_THRESHOLDS = [0.0, 0.08, 0.28, 0.50, 0.68, 0.82]
# PI = number of thresholds exceeded by the coverage ratio

def coverage_to_pi(coverage: float) -> int:
    """Convert plaque coverage ratio (0-1) to PI score (0-5)."""
    return sum(coverage > t for t in COVERAGE_THRESHOLDS)


def compute_plaque_coverage(image_rgb: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Full pipeline for one image: segment → detect → compute ratio.

    Returns:
        coverage  : float — plaque pixels / tooth pixels
        tooth_mask: (H,W) uint8
        plaque_mask: (H,W) uint8
    """
    tooth_mask  = segment_tooth_surface(image_rgb)
    tooth_area  = int(tooth_mask.sum() // 255)

    if tooth_area < 100:
        # Fallback: couldn't isolate tooth surface — return 0 (no plaque)
        return 0.0, tooth_mask, np.zeros_like(tooth_mask)

    plaque_mask  = detect_plaque_pixels(image_rgb, tooth_mask)
    plaque_area  = int(plaque_mask.sum() // 255)
    coverage     = plaque_area / tooth_area
    return coverage, tooth_mask, plaque_mask


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Multi-view PI Estimation with OHI Calibration
# ─────────────────────────────────────────────────────────────────────────────

def _ohi_prior_range(predicted_ohi: int) -> Tuple[float, float]:
    """
    OHI-S strongly correlates with PI. Use it as a soft prior.
    OHI 0 → patient unlikely to have PI > 2
    OHI 1 → PI likely 1–3
    OHI 2+ → PI likely 2–5
    """
    if predicted_ohi == 0:   return 0.0, 2.0
    elif predicted_ohi == 1: return 1.0, 3.5
    else:                     return 2.0, 5.0


def estimate_pi(
    frontal_rgb:   np.ndarray,
    left_rgb:      np.ndarray,
    right_rgb:     np.ndarray,
    predicted_ohi: int,
    verbose:       bool = False,
) -> dict:
    """
    Estimate Plaque Index (0–5) from three intra-oral views.

    Args:
        frontal_rgb   : uint8 RGB (H, W, 3) frontal view
        left_rgb      : uint8 RGB (H, W, 3) left lateral view
        right_rgb     : uint8 RGB (H, W, 3) right lateral view
        predicted_ohi : model-predicted OHI class (0, 1, 2)
        verbose       : if True, print intermediate values

    Returns:
        result dict:
            "pi_score"     : int (0–5) — final PI estimate
            "pi_raw"       : float — raw coverage-based score (unclipped)
            "coverage_f"   : float — frontal plaque coverage ratio
            "coverage_l"   : float — left coverage ratio
            "coverage_r"   : float — right coverage ratio
            "confidence"   : str — "high" / "medium" / "low"
            "tooth_area_px": int — frontal tooth area in pixels
    """
    coverages = []
    tooth_areas = []

    for view_img in (frontal_rgb, left_rgb, right_rgb):
        cov, t_mask, _ = compute_plaque_coverage(view_img)
        coverages.append(cov)
        tooth_areas.append(int(t_mask.sum() // 255))

    # Weight views by detected tooth area (more tooth visible → more reliable)
    areas   = np.array(tooth_areas, dtype=float)
    if areas.sum() < 1:
        # Complete fallback — no teeth detected in any view
        return {
            "pi_score": _ohi_fallback_pi(predicted_ohi),
            "pi_raw": float(_ohi_fallback_pi(predicted_ohi)),
            "coverage_f": 0.0, "coverage_l": 0.0, "coverage_r": 0.0,
            "confidence": "low (no tooth surface detected)",
            "tooth_area_px": 0,
        }

    weights   = areas / areas.sum()
    mean_cov  = float(np.dot(weights, coverages))
    raw_score = float(sum(mean_cov > t for t in COVERAGE_THRESHOLDS))

    # ── OHI calibration ────────────────────────────────────────────────────
    lo, hi  = _ohi_prior_range(predicted_ohi)
    # Soft clamp: blend raw_score toward the OHI-derived range
    # If raw_score is within [lo, hi], leave it unchanged.
    # If outside, pull it 40% toward the nearest boundary.
    blend_strength = 0.40
    if raw_score < lo:
        calibrated = raw_score + blend_strength * (lo - raw_score)
    elif raw_score > hi:
        calibrated = raw_score - blend_strength * (raw_score - hi)
    else:
        calibrated = raw_score

    pi_final = int(round(np.clip(calibrated, 0, 5)))

    # ── Confidence assessment ──────────────────────────────────────────────
    min_area = min(tooth_areas)
    if min_area > 3000 and max(coverages) - min(coverages) < 0.15:
        confidence = "high"
    elif min_area > 1000:
        confidence = "medium"
    else:
        confidence = "low (small tooth area detected)"

    if verbose:
        print(f"  Coverages  F={coverages[0]:.3f}  L={coverages[1]:.3f}  "
              f"R={coverages[2]:.3f}  weighted_mean={mean_cov:.3f}")
        print(f"  Raw PI={raw_score:.2f}  OHI prior=[{lo},{hi}]  "
              f"Calibrated={calibrated:.2f}  Final PI={pi_final}")

    return {
        "pi_score":     pi_final,
        "pi_raw":       round(calibrated, 2),
        "coverage_f":   round(coverages[0], 4),
        "coverage_l":   round(coverages[1], 4),
        "coverage_r":   round(coverages[2], 4),
        "confidence":   confidence,
        "tooth_area_px": tooth_areas[0],
    }


def _ohi_fallback_pi(ohi: int) -> int:
    """Pure OHI-based PI fallback when vision pipeline cannot detect teeth."""
    return {0: 0, 1: 1, 2: 2}.get(ohi, 1)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Django Integration — add to analysis/views.py
# ─────────────────────────────────────────────────────────────────────────────
#
# In your views.py, after you get model predictions for MGI/OHI/GEI:
#
#   from inference.pi_estimator import estimate_pi
#   import cv2
#
#   def run_analysis(frontal_path, left_path, right_path, ohi_prediction):
#       frontal = cv2.cvtColor(cv2.imread(frontal_path), cv2.COLOR_BGR2RGB)
#       left    = cv2.cvtColor(cv2.imread(left_path),    cv2.COLOR_BGR2RGB)
#       right   = cv2.cvtColor(cv2.imread(right_path),   cv2.COLOR_BGR2RGB)
#
#       pi_result = estimate_pi(
#           frontal_rgb=frontal,
#           left_rgb=left,
#           right_rgb=right,
#           predicted_ohi=ohi_prediction,   # int 0/1/2
#           verbose=False,
#       )
#
#       return {
#           "mgi": mgi_pred,
#           "ohi": ohi_pred,
#           "gei": gei_pred,
#           "pi":  pi_result["pi_score"],        # 0-5
#           "pi_confidence": pi_result["confidence"],
#       }
#
# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Visualisation helper (for debugging / Grad-CAM overlay integration)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_plaque_detection(image_rgb: np.ndarray,
                                save_path: Optional[str] = None) -> np.ndarray:
    """
    Returns annotated image with tooth (green overlay) and plaque (red overlay).
    Useful for debugging and for showing patients what was detected.
    """
    tooth_mask,  = (segment_tooth_surface(image_rgb),)
    plaque_mask  = detect_plaque_pixels(image_rgb, tooth_mask)

    overlay = image_rgb.copy()

    # Green: tooth (semi-transparent)
    tooth_color  = np.zeros_like(overlay)
    tooth_color[:, :] = [0, 220, 0]
    tooth_alpha  = (tooth_mask[:, :, None] / 255.0) * 0.25
    overlay      = (overlay * (1 - tooth_alpha) + tooth_color * tooth_alpha).astype(np.uint8)

    # Red: plaque
    plaque_color = np.zeros_like(overlay)
    plaque_color[:, :] = [255, 50, 50]
    plaque_alpha = (plaque_mask[:, :, None] / 255.0) * 0.55
    overlay      = (overlay * (1 - plaque_alpha) + plaque_color * plaque_alpha).astype(np.uint8)

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Calibration — run this on a representative sample to tune thresholds
# ─────────────────────────────────────────────────────────────────────────────
#
# If you later collect even 10 manually-scored PI photos, you can recalibrate:
#
#   from scipy.optimize import minimize
#
#   def calibrate_thresholds(images, true_pi_scores):
#       """
#       Fit COVERAGE_THRESHOLDS to match known PI scores.
#       Call this with a small calibration set.
#       """
#       coverages = [compute_plaque_coverage(img)[0] for img in images]
#
#       def loss(thresholds):
#           thresholds = sorted(thresholds)
#           preds = [sum(c > t for t in thresholds) for c in coverages]
#           return sum((p - y)**2 for p, y in zip(preds, true_pi_scores))
#
#       result = minimize(loss, COVERAGE_THRESHOLDS, method='Nelder-Mead')
#       return sorted(result.x.tolist())
#
#   new_thresholds = calibrate_thresholds(calib_images, calib_true_pi)
#   # Then update COVERAGE_THRESHOLDS with new_thresholds
#
# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST (run this file directly to test on a single image)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python pi_estimator.py <frontal.jpg> <left.jpg> <right.jpg> [ohi=0]")
        print("       OHI: 0=good, 1=fair, 2=poor")
        sys.exit(0)

    f_path = sys.argv[1]
    l_path = sys.argv[2]
    r_path = sys.argv[3]
    ohi    = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    def load(p):
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {p}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    frontal = load(f_path)
    left    = load(l_path)
    right   = load(r_path)

    result = estimate_pi(frontal, left, right, ohi, verbose=True)
    print(f"\n── PI Estimation Result ──────────────────")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")

    # Save debug overlay for frontal view
    out_path = "plaque_debug_frontal.jpg"
    visualize_plaque_detection(frontal, save_path=out_path)
    print(f"\nDebug overlay saved → {out_path}")
