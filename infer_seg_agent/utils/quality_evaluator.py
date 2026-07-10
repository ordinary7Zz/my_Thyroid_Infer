"""
Segmentation quality evaluation without ground truth
Evaluates mask quality based on morphological features and consistency
"""

import cv2
import numpy as np
from typing import Dict, List, Any
from .metrics import compute_pairwise_iou, compute_average_agreement, compute_hd95


class SegmentationQualityEvaluator:
    """
    Evaluate segmentation mask quality using unsupervised metrics
    """
    
    def __init__(self):
        """Initialize quality evaluator"""
        pass
    
    def evaluate_single_mask(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate quality of a single mask based on morphological features
        
        Args:
            mask: Binary mask (H, W) with values 0 or 1
            
        Returns:
            Dictionary with quality metrics
        """
        scores = {}
        
        # Basic statistics
        scores['area'] = int(np.sum(mask))
        scores['total_pixels'] = int(mask.size)
        scores['area_ratio'] = float(scores['area'] / scores['total_pixels'])
        
        # Find connected components
        num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        
        # Number of foreground components (excluding background)
        scores['num_components'] = int(num_components - 1)
        scores['is_single_component'] = (num_components == 2)
        
        # If there are multiple components, find the largest one
        if num_components > 2:
            # Get areas of all components except background (index 0)
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            largest_area = int(np.max(component_areas))
            scores['largest_component_area'] = largest_area
            scores['largest_component_ratio'] = largest_area / scores['area'] if scores['area'] > 0 else 0
        else:
            scores['largest_component_area'] = scores['area']
            scores['largest_component_ratio'] = 1.0
        
        # Find contours for shape analysis
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE
        )
        
        if contours:
            # Use the largest contour
            contour = max(contours, key=cv2.contourArea)
            
            # Perimeter and area from contour
            perimeter = cv2.arcLength(contour, True)
            contour_area = cv2.contourArea(contour)
            
            scores['perimeter'] = float(perimeter)
            scores['contour_area'] = float(contour_area)
            
            # Circularity: 4π*area/perimeter², closer to 1 means more circular
            if perimeter > 0:
                scores['circularity'] = float(4 * np.pi * contour_area / (perimeter ** 2))
            else:
                scores['circularity'] = 0.0
            
            # Compactness
            if perimeter > 0:
                scores['compactness'] = float(contour_area / (perimeter ** 2))
            else:
                scores['compactness'] = 0.0
            
            # Aspect ratio from bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            scores['bbox_width'] = int(w)
            scores['bbox_height'] = int(h)
            scores['aspect_ratio'] = float(w / h) if h > 0 else 0.0
            
            # Extent: ratio of contour area to bounding box area
            bbox_area = w * h
            scores['extent'] = float(contour_area / bbox_area) if bbox_area > 0 else 0.0
            
            # Solidity: ratio of contour area to convex hull area
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            scores['solidity'] = float(contour_area / hull_area) if hull_area > 0 else 0.0
            
            # Boundary smoothness
            scores['smoothness'] = self._compute_boundary_smoothness(contour)
            
        else:
            # Empty mask or no valid contours
            scores['perimeter'] = 0.0
            scores['contour_area'] = 0.0
            scores['circularity'] = 0.0
            scores['compactness'] = 0.0
            scores['bbox_width'] = 0
            scores['bbox_height'] = 0
            scores['aspect_ratio'] = 0.0
            scores['extent'] = 0.0
            scores['solidity'] = 0.0
            scores['smoothness'] = 0.0
        
        return scores
    
    def _compute_boundary_smoothness(self, contour: np.ndarray) -> float:
        """
        Compute boundary smoothness score
        Higher score means smoother boundary
        
        Args:
            contour: OpenCV contour
            
        Returns:
            Smoothness score (0-1, higher is smoother)
        """
        if len(contour) < 3:
            return 0.0
        
        # Compute angles between consecutive line segments
        angles = []
        n = len(contour)
        
        for i in range(n):
            p1 = contour[i - 1][0]
            p2 = contour[i][0]
            p3 = contour[(i + 1) % n][0]
            
            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Compute angle
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                # Clip to avoid numerical errors
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        if not angles:
            return 0.0
        
        # Standard deviation of angles - lower means smoother
        angle_std = np.std(angles)
        
        # Convert to 0-1 score (lower std = higher score)
        # Use sigmoid-like function
        smoothness = 1.0 / (1.0 + angle_std)
        
        return float(smoothness)
    
    def evaluate_model_agreement(self, masks: List[np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate agreement between multiple model outputs
        
        Args:
            masks: List of binary masks from different models
            
        Returns:
            Dictionary with agreement metrics
        """
        num_models = len(masks)
        if num_models < 2:
            return {
                'num_models': num_models,
                'pairwise_iou_matrix': None,
                'average_agreement': None,
                'overall_agreement': 0.0,
                'normalized_shape': masks[0].shape[:2] if masks else None,
                'volumes': None,
                'volume_mean': None,
                'volume_std': None,
                'volume_cv': None,
                'pairwise_hd95_matrix': None,
                'pairwise_hd95_mean': None,
                'pairwise_hd95_std': None,
            }
        
        # Normalize all masks to the same size for fair comparison
        # Use the first mask's size as the target size
        target_shape = masks[0].shape[:2]
        normalized_masks: List[np.ndarray] = []
        
        for mask in masks:
            if mask.shape[:2] != target_shape:
                normalized_mask = cv2.resize(
                    mask.astype(np.uint8),
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.uint8)
                normalized_masks.append(normalized_mask)
            else:
                normalized_masks.append(mask.astype(np.uint8))
        
        # Compute pairwise IoU on normalized masks
        iou_matrix = compute_pairwise_iou(normalized_masks)
        
        # Average agreement for each mask
        avg_agreement = compute_average_agreement(normalized_masks)
        
        # Overall agreement (mean of upper triangle, excluding diagonal)
        n = len(normalized_masks)
        upper_triangle = iou_matrix[np.triu_indices(n, k=1)]
        overall_agreement = float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else 0.0
        
        # --- Volume statistics across models (用于模型间面积 CV) ---
        volumes = np.array([int(m.sum()) for m in normalized_masks], dtype=np.float64)
        volume_mean = float(volumes.mean())
        volume_std = float(volumes.std(ddof=1)) if volumes.size > 1 else 0.0
        volume_cv = float(volume_std / volume_mean) if volume_mean > 0 else 0.0
        
        # --- Pairwise HD95 between model masks (用于边界差异估计) ---
        hd95_matrix = np.zeros((n, n), dtype=np.float32)
        hd95_values = []
        for i in range(n):
            for j in range(i + 1, n):
                hd = compute_hd95(normalized_masks[i], normalized_masks[j])
                hd95_matrix[i, j] = hd
                hd95_matrix[j, i] = hd
                hd95_values.append(hd)
        
        if hd95_values:
            hd_arr = np.array(hd95_values, dtype=np.float64)
            pairwise_hd95_mean = float(hd_arr.mean())
            pairwise_hd95_std = float(hd_arr.std(ddof=1)) if hd_arr.size > 1 else 0.0
        else:
            pairwise_hd95_mean = 0.0
            pairwise_hd95_std = 0.0
        
        return {
            'num_models': num_models,
            'pairwise_iou_matrix': iou_matrix.tolist(),
            'average_agreement': avg_agreement.tolist(),
            'overall_agreement': overall_agreement,
            'normalized_shape': target_shape,  # 归一化后的尺寸
            'volumes': volumes.tolist(),
            'volume_mean': volume_mean,
            'volume_std': volume_std,
            'volume_cv': volume_cv,
            'pairwise_hd95_matrix': hd95_matrix.tolist(),
            'pairwise_hd95_mean': pairwise_hd95_mean,
            'pairwise_hd95_std': pairwise_hd95_std,
        }
    
    def evaluate_batch(
        self, 
        masks: List[np.ndarray],
        model_names: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of masks from different models
        
        Args:
            masks: List of binary masks
            model_names: List of model names corresponding to masks
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        results = {
            'num_models': len(masks),
            'model_names': model_names,
            'individual_quality': [],
            'agreement_metrics': {}
        }
        
        # Evaluate each mask individually
        for mask, name in zip(masks, model_names):
            quality = self.evaluate_single_mask(mask)
            quality['model_name'] = name
            results['individual_quality'].append(quality)
        
        # Evaluate model agreement
        agreement = self.evaluate_model_agreement(masks)
        results['agreement_metrics'] = agreement
        
        return results
    
    def get_quality_summary(self, quality_metrics: Dict[str, Any]) -> str:
        """
        Generate human-readable quality summary
        
        Args:
            quality_metrics: Quality metrics from evaluate_single_mask
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Area
        summary_parts.append(f"面积: {quality_metrics['area']} 像素")
        
        # Connectivity
        if quality_metrics['is_single_component']:
            summary_parts.append("单连通区域")
        else:
            summary_parts.append(f"多连通区域 ({quality_metrics['num_components']} 个)")
        
        # Shape
        circularity = quality_metrics.get('circularity', 0)
        if circularity > 0.8:
            summary_parts.append("形状接近圆形")
        elif circularity > 0.6:
            summary_parts.append("形状较规则")
        else:
            summary_parts.append("形状不规则")
        
        # Smoothness
        smoothness = quality_metrics.get('smoothness', 0)
        if smoothness > 0.7:
            summary_parts.append("边界平滑")
        elif smoothness > 0.5:
            summary_parts.append("边界较平滑")
        else:
            summary_parts.append("边界粗糙")
        
        return ", ".join(summary_parts)

