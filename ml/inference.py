"""
Inference module for dental index prediction.
Loads a trained DINOv2 + ensemble model and predicts MGI, OHI, GEI scores.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path

torch = None

from ml.transforms import get_inference_transforms

_model_cache = None
_model_path_cache = None
_device = None


def _ensure_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch


def get_device():
    global _device
    if _device is None:
        torch_module = _ensure_torch()
        _device = torch_module.device('cuda' if torch_module.cuda.is_available() else 'cpu')
    return _device


def clear_model_cache():
    global _model_cache, _model_path_cache
    _model_cache = None
    _model_path_cache = None


def load_trained_model(checkpoint_path=None):
    """Load the trained model (cached for reuse)."""
    global _model_cache, _model_path_cache
    _ensure_torch()

    if checkpoint_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        checkpoint_path = str(base_dir / 'ml' / 'checkpoints' / 'best_model.pth')

    checkpoint_path = str(checkpoint_path)

    if _model_cache is not None and _model_path_cache == checkpoint_path:
        return _model_cache

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            f"Please train the model first using the Train_Model.ipynb notebook."
        )

    from ml.model import load_model
    device = get_device()
    model = load_model(checkpoint_path, device)
    _model_cache = model
    _model_path_cache = checkpoint_path
    print(f"Model loaded from {checkpoint_path} on {device}")
    return model


def predict_from_images(frontal_path, left_path, right_path, checkpoint_path=None):
    """
    Predict dental indices from 3 image file paths.

    Returns:
        dict with predictions:
            {
                'mgi': {'score': int, 'confidence': float},
                'ohi': {'score': int, 'confidence': float},
                'gei': {'score': int, 'confidence': float},
                'gradcam': dict of overlay PIL Images or None
            }
    """
    _ensure_torch()

    device = get_device()
    model = load_trained_model(checkpoint_path)
    transform = get_inference_transforms()

    frontal_pil = Image.open(frontal_path).convert('RGB')
    left_pil = Image.open(left_path).convert('RGB')
    right_pil = Image.open(right_path).convert('RGB')

    frontal_tensor = transform(frontal_pil).unsqueeze(0).to(device)
    left_tensor = transform(left_pil).unsqueeze(0).to(device)
    right_tensor = transform(right_pil).unsqueeze(0).to(device)

    results = model.predict_scores(frontal_tensor, left_tensor, right_tensor)

    predictions = {}
    for key in ['mgi', 'ohi', 'gei']:
        predictions[key] = {
            'score': results[key]['score'].item(),
            'confidence': results[key]['confidence'].item(),
        }

    # Grad-CAM (best effort — may fail for ensemble/ViT models)
    try:
        from ml.gradcam import generate_gradcam_for_patient
        images = {
            'frontal': frontal_pil,
            'left_lateral': left_pil,
            'right_lateral': right_pil,
        }
        gradcam_overlays = generate_gradcam_for_patient(model, images, device)
        predictions['gradcam'] = gradcam_overlays
    except Exception as e:
        print(f"Grad-CAM generation failed (expected for ViT models): {e}")
        predictions['gradcam'] = None

    return predictions


def predict_from_pil_images(frontal_pil, left_pil, right_pil, checkpoint_path=None):
    """Predict dental indices from 3 PIL Image objects."""
    _ensure_torch()

    device = get_device()
    model = load_trained_model(checkpoint_path)
    transform = get_inference_transforms()

    frontal_pil = frontal_pil.convert('RGB')
    left_pil = left_pil.convert('RGB')
    right_pil = right_pil.convert('RGB')

    frontal_tensor = transform(frontal_pil).unsqueeze(0).to(device)
    left_tensor = transform(left_pil).unsqueeze(0).to(device)
    right_tensor = transform(right_pil).unsqueeze(0).to(device)

    results = model.predict_scores(frontal_tensor, left_tensor, right_tensor)

    predictions = {}
    for key in ['mgi', 'ohi', 'gei']:
        predictions[key] = {
            'score': results[key]['score'].item(),
            'confidence': results[key]['confidence'].item(),
        }

    try:
        from ml.gradcam import generate_gradcam_for_patient
        images = {
            'frontal': frontal_pil,
            'left_lateral': left_pil,
            'right_lateral': right_pil,
        }
        gradcam_overlays = generate_gradcam_for_patient(model, images, device)
        predictions['gradcam'] = gradcam_overlays
    except Exception as e:
        print(f"Grad-CAM generation failed (expected for ViT models): {e}")
        predictions['gradcam'] = None

    return predictions
