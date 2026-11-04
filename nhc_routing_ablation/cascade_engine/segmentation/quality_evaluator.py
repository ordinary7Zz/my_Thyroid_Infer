"""
无 GT 的分割质量评估：形态学指标 + 跨模型一致性 + 分歧度量。

重写自 Segmentation_Agent/utils/quality_evaluator.py，逻辑一致。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .metrics import compute_pairwise_iou, compute_average_agreement, compute_hd95


class SegmentationQualityEvaluator:
    """基于形态学特征和模型间一致性的无监督分割质量评估。"""

    def evaluate_single_mask(self, mask: np.ndarray) -> dict[str, Any]:
        """
        评估单个 mask 的形态学质量。

        Args:
            mask: 二值 mask (H, W)，值为 0/1。

        Returns:
            含 area, num_components, circularity, solidity, smoothness 等指标的 dict。
        """
        mask_u8 = mask.astype(np.uint8)
        scores: dict[str, Any] = {}

        scores["area"] = int(np.sum(mask))
        scores["total_pixels"] = int(mask.size)
        scores["area_ratio"] = float(scores["area"] / scores["total_pixels"]) if scores["total_pixels"] > 0 else 0.0

        num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        scores["num_components"] = int(num_components - 1)
        scores["is_single_component"] = (num_components == 2)

        if num_components > 2:
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            largest = int(np.max(component_areas))
            scores["largest_component_area"] = largest
            scores["largest_component_ratio"] = largest / scores["area"] if scores["area"] > 0 else 0.0
        else:
            scores["largest_component_area"] = scores["area"]
            scores["largest_component_ratio"] = 1.0

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            contour_area = cv2.contourArea(contour)

            scores["perimeter"] = float(perimeter)
            scores["contour_area"] = float(contour_area)
            scores["circularity"] = (
                float(4 * np.pi * contour_area / (perimeter ** 2)) if perimeter > 0 else 0.0
            )
            scores["compactness"] = (
                float(contour_area / (perimeter ** 2)) if perimeter > 0 else 0.0
            )

            x, y, w, h = cv2.boundingRect(contour)
            scores["bbox_width"] = int(w)
            scores["bbox_height"] = int(h)
            scores["aspect_ratio"] = float(w / h) if h > 0 else 0.0

            bbox_area = w * h
            scores["extent"] = float(contour_area / bbox_area) if bbox_area > 0 else 0.0

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            scores["solidity"] = float(contour_area / hull_area) if hull_area > 0 else 0.0

            scores["smoothness"] = self._compute_boundary_smoothness(contour)
        else:
            for k in ["perimeter", "contour_area", "circularity", "compactness",
                       "aspect_ratio", "extent", "solidity", "smoothness"]:
                scores[k] = 0.0
            scores["bbox_width"] = 0
            scores["bbox_height"] = 0

        return scores

    def _compute_boundary_smoothness(self, contour: np.ndarray) -> float:
        """边界平滑度 (0-1, 越高越平滑)。基于连续线段夹角标准差。"""
        if len(contour) < 3:
            return 0.0

        n = len(contour)
        angles = []
        for i in range(n):
            p1 = contour[i - 1][0]
            p2 = contour[i][0]
            p3 = contour[(i + 1) % n][0]
            v1 = p2 - p1
            v2 = p3 - p2
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angles.append(np.arccos(cos_a))

        if not angles:
            return 0.0
        return float(1.0 / (1.0 + np.std(angles)))

    def evaluate_model_agreement(self, masks: list[np.ndarray]) -> dict[str, Any]:
        """
        评估多个模型输出之间的一致性。

        Returns:
            含 pairwise_iou_matrix, average_agreement, overall_agreement,
            volume_cv, pairwise_hd95_mean/std 的 dict。
        """
        n = len(masks)
        empty_result = {
            "num_models": n,
            "pairwise_iou_matrix": None,
            "average_agreement": None,
            "overall_agreement": 0.0,
            "normalized_shape": masks[0].shape[:2] if masks else None,
            "volumes": None,
            "volume_mean": None,
            "volume_std": None,
            "volume_cv": None,
            "pairwise_hd95_matrix": None,
            "pairwise_hd95_mean": None,
            "pairwise_hd95_std": None,
        }
        if n < 2:
            return empty_result

        target_shape = masks[0].shape[:2]
        normalized = []
        for mask in masks:
            if mask.shape[:2] != target_shape:
                m = cv2.resize(
                    mask.astype(np.uint8),
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.uint8)
            else:
                m = mask.astype(np.uint8)
            normalized.append(m)

        iou_matrix = compute_pairwise_iou(normalized)
        avg_agreement = compute_average_agreement(normalized)

        upper = iou_matrix[np.triu_indices(n, k=1)]
        overall = float(np.mean(upper)) if len(upper) > 0 else 0.0

        volumes = np.array([int(m.sum()) for m in normalized], dtype=np.float64)
        vol_mean = float(volumes.mean())
        vol_std = float(volumes.std(ddof=1)) if volumes.size > 1 else 0.0
        vol_cv = float(vol_std / vol_mean) if vol_mean > 0 else 0.0

        hd95_matrix = np.zeros((n, n), dtype=np.float32)
        hd95_values = []
        for i in range(n):
            for j in range(i + 1, n):
                hd = compute_hd95(normalized[i], normalized[j])
                hd95_matrix[i, j] = hd
                hd95_matrix[j, i] = hd
                hd95_values.append(hd)

        if hd95_values:
            hd_arr = np.array(hd95_values, dtype=np.float64)
            hd_mean = float(hd_arr.mean())
            hd_std = float(hd_arr.std(ddof=1)) if hd_arr.size > 1 else 0.0
        else:
            hd_mean = 0.0
            hd_std = 0.0

        return {
            "num_models": n,
            "pairwise_iou_matrix": iou_matrix.tolist(),
            "average_agreement": avg_agreement.tolist(),
            "overall_agreement": overall,
            "normalized_shape": target_shape,
            "volumes": volumes.tolist(),
            "volume_mean": vol_mean,
            "volume_std": vol_std,
            "volume_cv": vol_cv,
            "pairwise_hd95_matrix": hd95_matrix.tolist(),
            "pairwise_hd95_mean": hd_mean,
            "pairwise_hd95_std": hd_std,
        }

    def evaluate_batch(
        self,
        masks: list[np.ndarray],
        model_names: list[str],
    ) -> dict[str, Any]:
        """综合评估：每个 mask 的形态学 + 模型间一致性。"""
        results: dict[str, Any] = {
            "num_models": len(masks),
            "model_names": model_names,
            "individual_quality": [],
            "agreement_metrics": {},
        }
        for mask, name in zip(masks, model_names):
            q = self.evaluate_single_mask(mask)
            q["model_name"] = name
            results["individual_quality"].append(q)
        results["agreement_metrics"] = self.evaluate_model_agreement(masks)
        return results

    def get_quality_summary(self, quality_metrics: dict[str, Any]) -> str:
        """生成人类可读的质量摘要。"""
        parts = [f"面积: {quality_metrics['area']} 像素"]
        if quality_metrics.get("is_single_component"):
            parts.append("单连通区域")
        else:
            parts.append(f"多连通区域 ({quality_metrics.get('num_components', 0)} 个)")

        circ = quality_metrics.get("circularity", 0)
        if circ > 0.8:
            parts.append("形状接近圆形")
        elif circ > 0.6:
            parts.append("形状较规则")
        else:
            parts.append("形状不规则")

        smooth = quality_metrics.get("smoothness", 0)
        if smooth > 0.7:
            parts.append("边界平滑")
        elif smooth > 0.5:
            parts.append("边界较平滑")
        else:
            parts.append("边界粗糙")

        return ", ".join(parts)
