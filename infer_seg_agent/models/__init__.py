"""
Segmentation models module
"""

from .base_model import BaseSegmentationModel, ModelOutput
from .model_registry import ModelRegistry
from .dino_unet_model import DINOUNetSegmentationModel

__all__ = [
    'BaseSegmentationModel',
    'ModelOutput',
    'ModelRegistry',
    'DINOUNetSegmentationModel'
]

