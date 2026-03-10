"""
Inference module for dental index prediction.
Loads a trained model and predicts MGI, OHI, GEI scores from 3 dental photographs.
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from ml.model import load_model, MultiViewDentalModel
from ml.transforms import get_inference_transforms
from ml.gradcam import generate_gradcam_for_patient


# Global model cache
_model_cache = None
_device = None


def get_device():
    """Get the best available device."""
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device


def load_trained_model(checkpoint_path=None):
    """
    Load the trained model (cached for reuse).

    Args:
        checkpoint_path: Path to model checkpoint. If None, uses default path.

    Returns:
        Loaded model in eval mode.
    """
    global _model_cache

    if _model_cache is not None:
        return _model_cache

    if checkpoint_path is None:
        # Default checkpoint path
        base_dir = Path(__file__).resolve().parent.parent
        checkpoint_path = base_dir / 'ml' / 'checkpoints' / 'best_model.pth'

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            f"Please train the model first using: python ml/train.py"
        )

    device = get_device()
    model = load_model(checkpoint_path, device)
    _model_cache = model
    print(f"Model loaded from {checkpoint_path} on {device}")
    return model


def predict_from_images(frontal_path, left_path, right_path, checkpoint_path=None):
    """
    Predict dental indices from 3 image file paths.

    Args:
        frontal_path: Path to frontal photograph
        left_path: Path to left lateral photograph
        right_path: Path to right lateral photograph
        checkpoint_path: Optional path to model checkpoint

    Returns:
        dict with predictions:
            {
                'mgi': {'score': int, 'confidence': float},
                'ohi': {'score': int, 'confidence': float},
                'gei': {'score': int, 'confidence': float},
                'gradcam': {'frontal': PIL.Image, 'left_lateral': PIL.Image, 'right_lateral': PIL.Image}
            }
    """
    device = get_device()
    model = load_trained_model(checkpoint_path)
    transform = get_inference_transforms()

    # Load and transform images
    frontal_pil = Image.open(frontal_path).convert('RGB')
    left_pil = Image.open(left_path).convert('RGB')
    right_pil = Image.open(right_path).convert('RGB')

    frontal_tensor = transform(frontal_pil).unsqueeze(0).to(device)
    left_tensor = transform(left_pil).unsqueeze(0).to(device)
    right_tensor = transform(right_pil).unsqueeze(0).to(device)

    # Predict
    results = model.predict_scores(frontal_tensor, left_tensor, right_tensor)

    # Extract scores
    predictions = {}
    for key in ['mgi', 'ohi', 'gei']:
        predictions[key] = {
            'score': results[key]['score'].item(),
            'confidence': results[key]['confidence'].item(),
        }

    # Generate Grad-CAM overlays
    try:
        images = {
            'frontal': frontal_pil,
            'left_lateral': left_pil,
            'right_lateral': right_pil,
        }
        gradcam_overlays = generate_gradcam_for_patient(model, images, device)
        predictions['gradcam'] = gradcam_overlays
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        predictions['gradcam'] = None

    return predictions


def predict_from_pil_images(frontal_pil, left_pil, right_pil, checkpoint_path=None):
    """
    Predict dental indices from 3 PIL Image objects.
    Same as predict_from_images but accepts PIL Images directly.
    """
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
        images = {
            'frontal': frontal_pil,
            'left_lateral': left_pil,
            'right_lateral': right_pil,
        }
        gradcam_overlays = generate_gradcam_for_patient(model, images, device)
        predictions['gradcam'] = gradcam_overlays
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        predictions['gradcam'] = None

    return predictions
