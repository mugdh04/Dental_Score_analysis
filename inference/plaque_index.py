"""Rule-based plaque index computation from tooth image regions (Rewritten)."""

from __future__ import annotations
import json
import logging
from typing import Tuple, Dict, Any, List
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def prepare_tooth_for_pi(tooth_image: np.ndarray, tooth_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Isolate tooth and apply gray world white balance correction."""
    masked = cv2.bitwise_and(tooth_image, tooth_image, mask=tooth_mask)
    if tooth_mask.sum() == 0:
        return masked, np.array([])
        
    avg_r = masked[:, :, 0][tooth_mask > 0].mean()
    avg_g = masked[:, :, 1][tooth_mask > 0].mean()
    avg_b = masked[:, :, 2][tooth_mask > 0].mean()
    
    gray_avg = (avg_r + avg_g + avg_b) / 3.0
    if avg_r > 0 and avg_g > 0 and avg_b > 0:
        wb = masked.copy().astype(np.float32)
        wb[:, :, 0] = np.clip(wb[:, :, 0] * (gray_avg / avg_r), 0, 255)
        wb[:, :, 1] = np.clip(wb[:, :, 1] * (gray_avg / avg_g), 0, 255)
        wb[:, :, 2] = np.clip(wb[:, :, 2] * (gray_avg / avg_b), 0, 255)
        masked = wb.astype(np.uint8)
        
    pixels = masked[tooth_mask > 0]
    return masked, pixels

def detect_plaque_pixels(tooth_image: np.ndarray, tooth_mask: np.ndarray, gum_mask: np.ndarray = None) -> np.ndarray:
    """Creates 4 independent plaque masks and combines them."""
    if tooth_mask.sum() == 0:
        return np.zeros_like(tooth_mask)
        
    hsv = cv2.cvtColor(tooth_image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Mask A - HSV Yellow-Brown (primary plaque)
    mask_a = (h >= 10) & (h <= 42) & (s >= 30) & (v >= 60) & (v <= 230)
    
    # Mask B - LAB Yellow Channel (staining)
    lab = cv2.cvtColor(tooth_image, cv2.COLOR_RGB2LAB)
    l, a_chan, b_chan = cv2.split(lab)
    mask_b = (b_chan > 128) & (l >= 50) & (l <= 210)
    
    # Mask C - Relative Whiteness Deviation
    r, g, b = cv2.split(tooth_image)
    median_b = np.median(b[tooth_mask > 0])
    
    # Fast vectorized relative measurement
    r_dev = r.astype(np.int32) - median_b
    g_dev = g.astype(np.int32) - median_b
    mask_c = (r_dev > 15) & (g_dev > 8)
    
    # Mask D - Darkness Relative to Tooth
    median_v = np.median(v[tooth_mask > 0])
    mask_d = v < (median_v * 0.65)
    
    raw_mask = mask_a | mask_b | mask_c | mask_d
    
    # Exclude gums
    if gum_mask is not None:
        raw_mask = raw_mask & (~(gum_mask > 0))
        
    raw_mask = (raw_mask & (tooth_mask > 0)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    return closed

def compute_regional_pi(tooth_image: np.ndarray, tooth_mask: np.ndarray, plaque_mask: np.ndarray) -> float:
    """Divides tooth into 3 horizontal zones and computes weighted ratio."""
    coords = cv2.findNonZero((tooth_mask > 0).astype(np.uint8))
    if coords is None:
        return 0.0
        
    y_min, y_max = coords[:, 0, 1].min(), coords[:, 0, 1].max()
    h = y_max - y_min
    if h < 3:
        return 0.0
        
    # Assume 0 is top (gingival), H max is bottom (incisal)
    z1_end = y_min + h // 3
    z2_end = y_min + 2 * (h // 3)
    
    ratios = []
    zones = [(y_min, z1_end), (z1_end, z2_end), (z2_end, y_max)]
    
    for (start, end) in zones:
        zone_t = tooth_mask[start:end, :]
        zone_p = plaque_mask[start:end, :]
        t_pix = (zone_t > 0).sum()
        p_pix = (zone_p > 0).sum()
        ratios.append(p_pix / t_pix if t_pix > 0 else 0.0)
        
    # Zone 1 (Gingival Third) = 0.5, Zone 2 (Middle) = 0.3, Zone 3 (Incisal) = 0.2
    return (ratios[0] * 0.5) + (ratios[1] * 0.3) + (ratios[2] * 0.2)

def calibrate_pi_score(weighted_ratio: float, image_stats: dict, calibration_ref: dict) -> float:
    """Applies a correction factor based on full image stats."""
    if not calibration_ref:
        return min(max(weighted_ratio, 0.0), 1.0)
        
    # simple linear scaling based on v-channel median difference
    ref_b = calibration_ref.get("mean_brightness", 128.0)
    curr_b = image_stats.get("brightness", 128.0)
    
    # Prevent extreme scaling by clipping multiplier between 0.8 and 1.2
    multiplier = np.clip(curr_b / max(ref_b, 1.0), 0.8, 1.2)
    calibrated = weighted_ratio * multiplier
    
    return float(np.clip(calibrated, 0.0, 1.0))

def ratio_to_pi_score(calibrated_ratio: float) -> Tuple[int, str, float, str]:
    """Maps to 0-5 score."""
    if calibrated_ratio <= 0.05:
        score, label = 0, "No plaque"
    elif calibrated_ratio <= 0.15:
        score, label = 1, "Trace"
    elif calibrated_ratio <= 0.30:
        score, label = 2, "Mild"
    elif calibrated_ratio <= 0.50:
        score, label = 3, "Moderate"
    elif calibrated_ratio <= 0.70:
        score, label = 4, "Heavy"
    else:
        score, label = 5, "Severe"
        
    # Estimate confidence roughly based on distance to boundary
    boundaries = [0.00, 0.05, 0.15, 0.30, 0.50, 0.70, 1.00]
    conf = "high"
    for b in boundaries:
        if abs(calibrated_ratio - b) < 0.01:
            conf = "low"
        elif abs(calibrated_ratio - b) < 0.025:
            conf = "medium"
            
    return score, label, calibrated_ratio, conf

def aggregate_pi_for_image(per_tooth_scores: List[Tuple[float, int]]) -> Dict[str, Any]:
    """Computes mean calibrated ratio and variability."""
    if not per_tooth_scores:
        return {"score": 0, "ratio": 0.0, "label": "No plaque", "variability": 0.0}
        
    ratios = [r for r, s in per_tooth_scores]
    mean_ratio = float(np.mean(ratios))
    std_ratio = float(np.std(ratios))
    
    score, label, _, _ = ratio_to_pi_score(mean_ratio)
    return {
        "score": score,
        "ratio": mean_ratio,
        "label": label,
        "variability": std_ratio
    }

def aggregate_pi_across_views(frontal_pi: Dict, left_pi: Dict, right_pi: Dict) -> Dict[str, Any]:
    """Weights views: frontal x 0.4, left x 0.3, right x 0.3."""
    
    f_ratio = frontal_pi.get("ratio", 0.0)
    l_ratio = left_pi.get("ratio", 0.0)
    r_ratio = right_pi.get("ratio", 0.0)
    
    weighted_mean = (f_ratio * 0.4) + (l_ratio * 0.3) + (r_ratio * 0.3)
    score, label, ratio, conf = ratio_to_pi_score(weighted_mean)
    
    f_var = frontal_pi.get("variability", 0.0)
    l_var = left_pi.get("variability", 0.0)
    r_var = right_pi.get("variability", 0.0)
    
    overall_var = np.mean([f_var, l_var, r_var])
    
    return {
        "pi_score": score,
        "pi_ratio": ratio,
        "pi_label": label,
        "pi_variability": overall_var,
        "pi_confidence": conf,
        "per_view": {
            "frontal": {"score": frontal_pi.get("score", 0), "ratio": f_ratio},
            "left": {"score": left_pi.get("score", 0), "ratio": l_ratio},
            "right": {"score": right_pi.get("score", 0), "ratio": r_ratio}
        }
    }
