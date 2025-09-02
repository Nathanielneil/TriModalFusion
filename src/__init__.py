"""
TriModalFusion: A unified multimodal recognition system for speech, gesture, and image processing.

This package provides a comprehensive framework for multimodal learning that integrates:
- Speech recognition (based on Transformer architectures)
- Gesture recognition (using MediaPipe + Graph Convolutional Networks)
- Image recognition and detection (using Vision Transformers and YOLO)
- Cross-modal attention mechanisms for feature fusion

Author: AI Research Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "research@trimodal.ai"

from .models import TriModalFusionModel
from .encoders import SpeechEncoder, GestureEncoder, ImageEncoder
from .fusion import CrossModalFusion, HierarchicalFusion
from .utils import load_config, setup_logging

__all__ = [
    "TriModalFusionModel",
    "SpeechEncoder", 
    "GestureEncoder", 
    "ImageEncoder",
    "CrossModalFusion",
    "HierarchicalFusion",
    "load_config",
    "setup_logging"
]