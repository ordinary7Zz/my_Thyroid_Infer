"""
Base class for segmentation models
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class ModelOutput:
    """
    Output from a segmentation model
    """
    model_name: str
    mask: np.ndarray  # Binary mask (H, W) with values 0 or 1
    confidence_map: Optional[np.ndarray] = None  # Optional probability map (H, W) with values 0-1
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata (e.g., training devices)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)"""
        result = {
            'model_name': self.model_name,
            'mask_shape': self.mask.shape,
            'mask_area': int(np.sum(self.mask)),
            'metadata': self.metadata or {}
        }
        
        if self.confidence_map is not None:
            result['has_confidence_map'] = True
            result['mean_confidence'] = float(np.mean(self.confidence_map[self.mask > 0])) if np.sum(self.mask) > 0 else 0.0
        else:
            result['has_confidence_map'] = False
        
        return result


class BaseSegmentationModel(ABC):
    """
    Abstract base class for segmentation models
    All segmentation models should inherit from this class
    """
    
    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        device: str = 'cuda',
        training_data_devices: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the segmentation model
        
        Args:
            model_name: Name of the model
            model_path: Path to model weights
            device: Device to run model on ('cuda' or 'cpu')
            training_data_devices: List of devices used in training data (e.g., ["GE Logiq E9", "Siemens"])
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.training_data_devices = training_data_devices or []
        self.model = None
        self.is_loaded = False
        self.metadata = kwargs
        
        # Load model if path is provided
        if model_path:
            self.load_model()
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load model from file
        Must set self.model and self.is_loaded = True
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Any:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (H, W, 3) in RGB format, float32, range [0, 1]
            
        Returns:
            Preprocessed input for the model
        """
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> ModelOutput:
        """
        Run segmentation on the input image
        
        Args:
            image: Input image (H, W, 3) in RGB format, float32, range [0, 1]
            
        Returns:
            ModelOutput with segmentation mask
        """
        pass
    
    def postprocess(
        self, 
        output: Any, 
        original_shape: tuple,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Postprocess model output to binary mask
        
        Args:
            output: Raw model output
            original_shape: Original image shape (H, W)
            threshold: Threshold for binarization
            
        Returns:
            Binary mask (H, W) with values 0 or 1
        """
        # Default implementation - override if needed
        # Assumes output is a probability map
        if isinstance(output, np.ndarray):
            # Remove extra dimensions
            while output.ndim > 2:
                output = output.squeeze(0)
            
            # Threshold
            mask = (output > threshold).astype(np.uint8)
            
            # Resize if needed
            if mask.shape != original_shape:
                import cv2
                mask = cv2.resize(mask, (original_shape[1], original_shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            return mask
        
        raise NotImplementedError("postprocess must handle the specific output format")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'device': self.device,
            'training_data_devices': self.training_data_devices,
            'is_loaded': self.is_loaded,
            **self.metadata
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name}, loaded={self.is_loaded})"

