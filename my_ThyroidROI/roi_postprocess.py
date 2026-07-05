import cv2
import numpy as np
from typing import Tuple, Optional, List
import warnings
from pathlib import Path


class ROIPostProcessor:
    """
    ROI掩码后处理器
    - 平滑边界
    - 提取最大连通域
    - 拟合规则形状
    - 裁剪原图
    """

    def __init__(self,
                 morph_kernel_size: int = 5,
                 close_iterations: int = 2,
                 open_iterations: int = 1,
                 min_area_ratio: float = 0.01):
        self.morph_kernel_size = morph_kernel_size
        self.close_iterations = close_iterations
        self.open_iterations = open_iterations
        self.min_area_ratio = min_area_ratio

        # 缓存卷积核
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )

    def smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        """平滑掩码边界"""
        mask = (mask > 0).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel,
                                iterations=self.close_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel,
                                iterations=self.open_iterations)
        return (mask > 0).astype(np.uint8) * 255

    def get_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """提取最大连通域"""
        mask_bin = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        if num_labels <= 1:
            return (mask_bin * 255).astype(np.uint8)

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1
        total_area = mask_bin.shape[0] * mask_bin.shape[1]
        if areas[largest_idx - 1] < total_area * self.min_area_ratio:
            warnings.warn("最大连通域面积过小，可能分割失败")

        largest_mask = (labels == largest_idx).astype(np.uint8) * 255
        return largest_mask

    def fit_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 0, 0, mask.shape[1], mask.shape[0]
        all_points = np.vstack(contours)
        return cv2.boundingRect(all_points)

    def fit_rotated_rect(self, mask: np.ndarray) -> Tuple[Tuple, np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            h, w = mask.shape
            return ((w / 2, h / 2), (w, h), 0), np.array([[0, 0], [w, 0], [w, h], [0, h]])
        all_points = np.vstack(contours)
        rect = cv2.minAreaRect(all_points)
        box = cv2.boxPoints(rect).astype(int)
        return rect, box

    def fit_convex_hull(self, mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            h, w = mask.shape
            return np.array([[0, 0], [w, 0], [w, h], [0, h]])
        all_points = np.vstack(contours)
        return cv2.convexHull(all_points)

    def fit_polygon(self, mask: np.ndarray, epsilon_ratio: float = 0.01) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            h, w = mask.shape
            return np.array([[0, 0], [w, 0], [w, h], [0, h]])
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = max(epsilon_ratio * max(perimeter, 1.0), 1.0)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        return approx.reshape(-1, 2)

    def create_shape_mask(self,
                          shape: str,
                          mask_shape: Tuple[int, int],
                          shape_params) -> np.ndarray:
        """根据拟合形状创建规则掩码"""
        h, w = mask_shape
        new_mask = np.zeros((h, w), dtype=np.uint8)

        if shape == 'bbox':
            x, y, box_w, box_h = shape_params
            cv2.rectangle(new_mask, (x, y), (x + box_w, y + box_h), 255, -1)
        elif shape == 'rotated_rect':
            _, box_points = shape_params
            cv2.fillPoly(new_mask, [box_points.astype(np.int32)], 255)
        elif shape == 'convex_hull':
            cv2.fillPoly(new_mask, [shape_params.astype(np.int32)], 255)
        elif shape == 'polygon':
            cv2.fillPoly(new_mask, [shape_params.astype(np.int32)], 255)
        return new_mask

    def process_mask(self,
                     mask: np.ndarray,
                     shape_type: str = 'rotated_rect') -> Tuple[np.ndarray, dict]:
        """完整掩码处理流程"""
        info = {}
        smoothed = self.smooth_mask(mask)
        info['smoothed'] = smoothed
        largest = self.get_largest_component(smoothed)
        info['largest_component'] = largest

        if shape_type == 'smooth':
            processed = largest
            info['shape_type'] = 'smooth'
        elif shape_type == 'bbox':
            bbox = self.fit_bounding_box(largest)
            processed = self.create_shape_mask('bbox', mask.shape, bbox)
            info.update({'shape_type': 'bbox', 'bbox': bbox})
        elif shape_type == 'rotated_rect':
            rect, box = self.fit_rotated_rect(largest)
            processed = self.create_shape_mask('rotated_rect', mask.shape, (rect, box))
            info.update({'shape_type': 'rotated_rect', 'rect': rect, 'box_points': box})
        elif shape_type == 'convex_hull':
            hull = self.fit_convex_hull(largest)
            processed = self.create_shape_mask('convex_hull', mask.shape, hull)
            info.update({'shape_type': 'convex_hull', 'hull': hull})
        elif shape_type == 'polygon':
            poly = self.fit_polygon(largest, epsilon_ratio=0.01)
            processed = self.create_shape_mask('polygon', mask.shape, poly)
            info.update({'shape_type': 'polygon', 'polygon': poly})
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
        return processed, info

    def crop_image(self,
                   image: np.ndarray,
                   mask: np.ndarray,
                   padding: int = 0,
                   background_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """根据掩码裁剪图像"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return image.copy()
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        cropped = image[y:y + h, x:x + w].copy()
        cropped_mask = (mask[y:y + h, x:x + w] > 0)
        if cropped.ndim == 3:
            cropped[~cropped_mask] = background_color
        else:
            cropped = np.where(cropped_mask, cropped, background_color[0]).astype(cropped.dtype)
        return cropped


def crop_rotated_rect(image: np.ndarray, rect, bg=(0, 0, 0)) -> np.ndarray:
    """旋转对齐裁剪（可选增强）"""
    (cx, cy), (w, h), angle = rect
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    w_i, h_i = int(round(w)), int(round(h))
    x, y = int(cx - w_i / 2), int(cy - h_i / 2)
    x, y = max(0, x), max(0, y)
    x2, y2 = min(rotated.shape[1], x + w_i), min(rotated.shape[0], y + h_i)
    return rotated[y:y2, x:x2]


def crop_roi_from_image(image_path: str,
                        mask: np.ndarray,
                        output_path: str,
                        shape_type: str = 'rotated_rect',
                        padding: int = 5,
                        save_visualization: bool = True) -> dict:
    """从图像中裁剪ROI区域完整流程"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    processor = ROIPostProcessor(
        morph_kernel_size=5, close_iterations=3, open_iterations=2, min_area_ratio=0.01
    )
    processed_mask, info = processor.process_mask(mask, shape_type=shape_type)

    if shape_type == 'rotated_rect' and 'rect' in info:
        cropped_image = crop_rotated_rect(image, info['rect'])
    else:
        cropped_image = processor.crop_image(image, processed_mask, padding=padding)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(output_path, cropped_image)
    if not ok:
        raise IOError(f"保存失败: {output_path}")

    if save_visualization:
        vis_path = output_path.replace('.png', '_comparison.png')
        save_comparison_visualization(image, mask, processed_mask, cropped_image, vis_path, info)

    return {
        'input_image': image_path,
        'output_image': output_path,
        'original_size': image.shape[:2],
        'cropped_size': cropped_image.shape[:2],
        'shape_type': shape_type,
        'processing_info': info
    }


def save_comparison_visualization(original_image: np.ndarray,
                                  original_mask: np.ndarray,
                                  processed_mask: np.ndarray,
                                  cropped_image: np.ndarray,
                                  output_path: str,
                                  info: dict):
    """保存对比可视化图"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('1. 原始图像', fontsize=12, fontweight='bold'); axes[0, 0].axis('off')
    axes[0, 1].imshow(original_mask, cmap='gray')
    axes[0, 1].set_title('2. 分割模型输出掩码', fontsize=12, fontweight='bold'); axes[0, 1].axis('off')
    if 'smoothed' in info:
        axes[0, 2].imshow(info['smoothed'], cmap='gray')
        axes[0, 2].set_title('3. 形态学平滑', fontsize=12, fontweight='bold'); axes[0, 2].axis('off')
    if 'largest_component' in info:
        axes[1, 0].imshow(info['largest_component'], cmap='gray')
        axes[1, 0].set_title('4. 最大连通域', fontsize=12, fontweight='bold'); axes[1, 0].axis('off')

    axes[1, 1].imshow(processed_mask, cmap='gray')
    shape_name = info.get('shape_type', 'unknown')
    axes[1, 1].set_title(f'5. 规则化掩码 ({shape_name})', fontsize=12, fontweight='bold'); axes[1, 1].axis('off')
    axes[1, 2].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('6. 最终裁剪结果', fontsize=12, fontweight='bold'); axes[1, 2].axis('off')

    plt.suptitle('ROI提取完整流程', fontsize=16, fontweight='bold')
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'可视化对比图已保存: {output_path}')


def compare_shape_types(image: np.ndarray,
                        mask: np.ndarray,
                        output_path: str):
    """比较不同形状拟合方法的效果"""
    import matplotlib.pyplot as plt
    processor = ROIPostProcessor()
    shape_types = ['smooth', 'bbox', 'rotated_rect', 'convex_hull', 'polygon']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('原始图像', fontsize=12, fontweight='bold'); axes[0].axis('off')

    for idx, shape_type in enumerate(shape_types, 1):
        processed_mask, _ = processor.process_mask(mask, shape_type=shape_type)
        cropped = processor.crop_image(image, processed_mask, padding=5)
        axes[idx].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(f'{shape_type} {cropped.shape[1]}x{cropped.shape[0]}', fontsize=12, fontweight='bold')
        axes[idx].axis('off')

    plt.suptitle('不同形状拟合方法对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'形状对比图已保存: {output_path}')
