"""Prediction heads for CasCrop."""

from .waste_classifier import WasteClassifier
from .cause_classifier import CauseClassifier, CauseOfLoss

__all__ = [
    "WasteClassifier",
    "CauseClassifier",
    "CauseOfLoss",
]
