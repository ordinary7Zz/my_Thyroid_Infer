"""
Image and mask processing utilities
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional


class ImageProcessor:
    """
    Image and mask processing for segmentation tasks
    """
    
    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        """
        Initialize image processor
        
        Args:
            target_size: Target size (height, width) for resizing, None to keep original
            normalize: Whether to normalize images
            normalize_mean: Mean values for normalization (RGB)
            normalize_std: Std values for normalization (RGB)
        """
        self.target_size = target_size
        self.normalize = normalize
        self.normalize_mean = np.array(normalize_mean).reshape(1, 1, 3)
        self.normalize_std = np.array(normalize_std).reshape(1, 1, 3)
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image array (H, W, 3) in RGB format, normalized to [0, 1]
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image (OpenCV loads in BGR)
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to float [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Resize if needed (only for display/GT mask alignment, not for model input)
        # 注意：模型的预处理会统一处理 resize，这里只用于 GT mask 对齐
        if self.target_size is not None:
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        
        # 不再在这里进行归一化，归一化由模型的预处理统一处理
        # 返回 [0, 1] 范围的图像，模型会统一进行预处理
        
        return image
    
    def load_mask(self, mask_path: Union[str, Path]) -> np.ndarray:
        """
        Load binary mask from file
        
        Args:
            mask_path: Path to mask file
            
        Returns:
            Binary mask array (H, W) with values 0 or 1
        """
        mask_path = Path(mask_path)
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # Load mask as grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # Resize if needed
        if self.target_size is not None:
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Binarize (threshold at 127)
        mask = (mask > 127).astype(np.uint8)
        
        return mask
    
    def save_mask(self, mask: np.ndarray, save_path: Union[str, Path]) -> None:
        """
        Save binary mask to file
        
        Args:
            mask: Binary mask (H, W) with values 0 or 1
            save_path: Path to save mask
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to 0-255 range
        mask_img = (mask * 255).astype(np.uint8)
        
        # Save
        cv2.imwrite(str(save_path), mask_img)
    
    def preprocess_for_model(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input
        Typically converts (H, W, 3) to (3, H, W) or (1, 3, H, W)
        
        Args:
            image: Image array (H, W, 3)
            
        Returns:
            Preprocessed image (3, H, W)
        """
        # Transpose to (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def postprocess_mask(self, mask_logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Postprocess model output to binary mask
        
        Args:
            mask_logits: Model output, can be:
                - (H, W): Single channel probability map
                - (1, H, W): Single channel with batch dim
                - (1, 1, H, W): With batch and channel dims
            threshold: Threshold for binarization
            
        Returns:
            Binary mask (H, W) with values 0 or 1
        """
        # Remove extra dimensions
        while mask_logits.ndim > 2:
            mask_logits = mask_logits.squeeze(0)
        
        # Apply threshold
        mask = (mask_logits > threshold).astype(np.uint8)
        
        return mask
    
    def resize_mask_to_original(
        self, 
        mask: np.ndarray, 
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize mask back to original image size
        
        Args:
            mask: Binary mask (H, W)
            original_size: Original size (height, width)
            
        Returns:
            Resized binary mask
        """
        mask = cv2.resize(
            mask.astype(np.uint8), 
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        return mask

