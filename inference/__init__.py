"""Inference package for oral health prediction and plaque index scoring."""

from .plaque_index import compute_plaque_index, plaque_score_to_label
from .predict import OralHealthPredictor

__all__ = ["compute_plaque_index", "plaque_score_to_label", "OralHealthPredictor"]
