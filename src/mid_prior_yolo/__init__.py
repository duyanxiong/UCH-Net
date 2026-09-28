"""Minimal public implementation of the YOLO11-MM Mid-Prior detector."""

from .model import MidPriorDetectionModel
from .trainer import MidPriorTrainer
from .weights import load_public_weights

__all__ = ("MidPriorDetectionModel", "MidPriorTrainer", "load_public_weights")
__version__ = "0.1.0"

