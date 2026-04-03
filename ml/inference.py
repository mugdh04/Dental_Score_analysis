"""
Inference module for dental index prediction.
Loads a trained DINOv2 + ensemble model and predicts MGI, OHI, GEI scores.
Supports TTA (Test-Time Augmentation) for improved accuracy.
"""

# -----------------------------------------------------------------------------
# Change Note (2026-04-03)
# Added configurable model-path resolution using MODEL_PATH with safe fallbacks
# and switched startup/runtime diagnostics to logging.
# -----------------------------------------------------------------------------

import os
import logging
import numpy as np
from PIL import Image
from pathlib import Path

torch = None

from ml.transforms import get_inference_transforms, get_tta_transforms

_model_cache = None
_model_path_cache = None
_device = None
logger = logging.getLogger(__name__)


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


def _resolve_default_checkpoint_path() -> str:
    """Resolve checkpoint path from env/settings with compatibility fallback."""
    base_dir = Path(__file__).resolve().parent.parent
    env_model_path = os.environ.get('MODEL_PATH')

    if env_model_path:
        env_path = Path(env_model_path)
        if not env_path.is_absolute():
            env_path = (base_dir / env_path).resolve()
        if env_path.exists():
            return str(env_path)
        logger.warning('MODEL_PATH is set but file does not exist: %s', env_path)

    # Prefer Django settings path when available.
    try:
        from django.conf import settings as django_settings

        settings_path = getattr(django_settings, 'MODEL_PATH', None)
        if settings_path and Path(settings_path).exists():
            return str(settings_path)
    except Exception:
        pass

    models_default = base_dir / 'models' / 'multitask_model.pth'
    legacy_default = base_dir / 'ml' / 'checkpoints' / 'best_model.pth'

    if models_default.exists():
        return str(models_default)
    return str(legacy_default)


def load_trained_model(checkpoint_path=None):
    """Load the trained model (cached for reuse)."""
    global _model_cache, _model_path_cache
    _ensure_torch()

    if checkpoint_path is None:
        checkpoint_path = _resolve_default_checkpoint_path()

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
    logger.info('Model loaded from %s on %s', checkpoint_path, device)
    return model


def _predict_with_tta(model, frontal_pil, left_pil, right_pil, device):
    """Run TTA: average predictions over multiple augmented views."""
    # Get image size from model (either DINOv2MultiViewModel.img_size or EnsembleDentalModel)
    img_size = getattr(model, 'img_size', 336)
    tta_transforms = get_tta_transforms(image_size=img_size)
    all_probs = {k: [] for k in ['mgi', 'ohi', 'gei']}

    for tfm in tta_transforms:
        f_t = tfm(frontal_pil).unsqueeze(0).to(device)
        l_t = tfm(left_pil).unsqueeze(0).to(device)
        r_t = tfm(right_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            results = model.predict_scores(f_t, l_t, r_t)
        for key in ['mgi', 'ohi', 'gei']:
            all_probs[key].append(results[key]['probs'].cpu())

    predictions = {}
    for key in ['mgi', 'ohi', 'gei']:
        avg_probs = torch.stack(all_probs[key]).mean(dim=0)
        predicted = avg_probs.argmax(dim=1).int()
        confidence = avg_probs.max(dim=1).values * 100.0
        predictions[key] = {
            'score': predicted.item(),
            'confidence': confidence.item(),
        }
    return predictions


def _predict_standard(model, frontal_pil, left_pil, right_pil, device):
    """Standard single-pass prediction."""
    img_size = getattr(model, 'img_size', 336)
    transform = get_inference_transforms(image_size=img_size)
    f_t = transform(frontal_pil).unsqueeze(0).to(device)
    l_t = transform(left_pil).unsqueeze(0).to(device)
    r_t = transform(right_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        results = model.predict_scores(f_t, l_t, r_t)

    predictions = {}
    for key in ['mgi', 'ohi', 'gei']:
        predictions[key] = {
            'score': results[key]['score'].item(),
            'confidence': results[key]['confidence'].item(),
        }
    return predictions


def _attach_gradcam(predictions, model, frontal_pil, left_pil, right_pil, device):
    """Attempt Grad-CAM generation (best effort)."""
    try:
        from ml.gradcam import generate_gradcam_for_patient
        images = {
            'frontal': frontal_pil,
            'left_lateral': left_pil,
            'right_lateral': right_pil,
        }
        predictions['gradcam'] = generate_gradcam_for_patient(model, images, device)
    except Exception:
        predictions['gradcam'] = None
    return predictions


def predict_from_images(frontal_path, left_path, right_path, checkpoint_path=None, use_tta=True):
    """
    Predict dental indices from 3 image file paths.

    Args:
        use_tta: If True, use Test-Time Augmentation for better accuracy.

    Returns:
        dict with 'mgi', 'ohi', 'gei' predictions and optional 'gradcam'.
    """
    _ensure_torch()
    device = get_device()
    model = load_trained_model(checkpoint_path)

    frontal_pil = Image.open(frontal_path).convert('RGB')
    left_pil = Image.open(left_path).convert('RGB')
    right_pil = Image.open(right_path).convert('RGB')

    if use_tta:
        predictions = _predict_with_tta(model, frontal_pil, left_pil, right_pil, device)
    else:
        predictions = _predict_standard(model, frontal_pil, left_pil, right_pil, device)

    return _attach_gradcam(predictions, model, frontal_pil, left_pil, right_pil, device)


def predict_from_pil_images(frontal_pil, left_pil, right_pil, checkpoint_path=None, use_tta=True):
    """Predict dental indices from 3 PIL Image objects."""
    _ensure_torch()
    device = get_device()
    model = load_trained_model(checkpoint_path)

    frontal_pil = frontal_pil.convert('RGB')
    left_pil = left_pil.convert('RGB')
    right_pil = right_pil.convert('RGB')

    if use_tta:
        predictions = _predict_with_tta(model, frontal_pil, left_pil, right_pil, device)
    else:
        predictions = _predict_standard(model, frontal_pil, left_pil, right_pil, device)

    return _attach_gradcam(predictions, model, frontal_pil, left_pil, right_pil, device)
