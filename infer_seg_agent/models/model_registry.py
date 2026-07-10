"""
Model registry for managing multiple segmentation models
"""

import numpy as np
from typing import List, Dict, Optional
from .base_model import BaseSegmentationModel, ModelOutput


class ModelRegistry:
    """
    Registry for managing multiple segmentation models
    """
    
    def __init__(self):
        """Initialize empty model registry"""
        self.models: List[BaseSegmentationModel] = []
        self.model_names: List[str] = []
    
    def register_model(self, model: BaseSegmentationModel) -> None:
        """
        Register a new model
        
        Args:
            model: Instance of BaseSegmentationModel
        """
        if not isinstance(model, BaseSegmentationModel):
            raise TypeError(f"Model must be instance of BaseSegmentationModel, got {type(model)}")
        
        if model.model_name in self.model_names:
            raise ValueError(f"Model with name '{model.model_name}' already registered")
        
        self.models.append(model)
        self.model_names.append(model.model_name)
        
        print(f"✓ Registered model: {model.model_name}")
    
    def unregister_model(self, model_name: str) -> None:
        """
        Unregister a model by name
        
        Args:
            model_name: Name of the model to remove
        """
        if model_name not in self.model_names:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        idx = self.model_names.index(model_name)
        self.models.pop(idx)
        self.model_names.pop(idx)
        
        print(f"✓ Unregistered model: {model_name}")
    
    def get_model(self, model_name: str) -> Optional[BaseSegmentationModel]:
        """
        Get a model by name
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance or None if not found
        """
        if model_name not in self.model_names:
            return None
        
        idx = self.model_names.index(model_name)
        return self.models[idx]
    
    def predict_all(self, image: np.ndarray) -> List[ModelOutput]:
        """
        Run all registered models on the input image
        
        Args:
            image: Input image (H, W, 3) in RGB format, float32, range [0, 1]
            
        Returns:
            List of ModelOutput from all models
        """
        if not self.models:
            raise RuntimeError("No models registered in the registry")
        
        predictions = []
        
        for model in self.models:
            try:
                output = model.predict(image)
                predictions.append(output)
            except Exception as e:
                print(f"✗ Error running model {model.model_name}: {e}")
                # Continue with other models
                continue
        
        return predictions
    
    def list_models(self) -> List[str]:
        """
        Get list of registered model names
        
        Returns:
            List of model names
        """
        return self.model_names.copy()
    
    def get_model_info(self) -> List[Dict]:
        """
        Get information about all registered models
        
        Returns:
            List of model metadata dictionaries
        """
        return [model.get_metadata() for model in self.models]
    
    def __len__(self) -> int:
        """Return number of registered models"""
        return len(self.models)
    
    def __repr__(self) -> str:
        return f"ModelRegistry(num_models={len(self.models)}, models={self.model_names})"

