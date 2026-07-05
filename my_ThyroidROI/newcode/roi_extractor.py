# /data2/chenshiyu/ThyroidROI/roi_extractor.py
"""
ROIExtractor: Swin-UNet 模型的甲状腺超声 ROI 提取器
默认使用 rotated_rect 形状后处理
"""

import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional
from swin_unet import SwinUNet
from roi_postprocess import ROIPostProcessor

class ROIExtractor:

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        img_size: int = 224,
        threshold: float = 0.5,
        padding: int = 5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.threshold = threshold
        self.padding = padding

        self.model = self._load_model(checkpoint_path)
        self.transform = self._build_transform(img_size)

        # 默认使用 rotated_rect 后处理
        self.postprocessor = ROIPostProcessor(
            morph_kernel_size=5,
            close_iterations=3,
            open_iterations=2,
            min_area_ratio=0.01
        )

        print(f"ROIExtractor 初始化完成 (设备: {self.device})")

    def _build_transform(self, img_size: int):
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def _load_model(self, checkpoint_path: str):
        """加载训练好的 SwinUNet 模型"""
        model = SwinUNet(
            img_size=self.img_size,
            patch_size=4,
            in_chans=3,
            num_classes=1,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device).eval()
        print(f" 加载模型: {checkpoint_path}")
        print(f"  epoch={checkpoint.get('epoch','?')}  best_dice={checkpoint.get('best_dice','?')}")
        return model

    @torch.no_grad()
    def _predict_mask(self, image: np.ndarray) -> np.ndarray:
        """预测分割掩码 (返回0-1 mask)"""
        h, w = image.shape[:2]
        tensor = self.transform(image=image)["image"].unsqueeze(0).to(self.device)
        prob = torch.sigmoid(self.model(tensor)).squeeze().cpu().numpy()
        prob_resized = cv2.resize(prob, (w, h))
        mask = (prob_resized > self.threshold).astype(np.uint8)
        return mask

    def extract_roi(self, image_path: str) -> np.ndarray:
        """
        从输入图像中提取ROI区域。

        参数:
            image_path: 输入图像路径
        返回:
            roi_rgb: numpy.ndarray, 形状(H,W,3), RGB顺序, 值范围[0,1]
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像不存在: {image_path}")

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"无法读取图像: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = self._predict_mask(image_rgb)

        processed_mask, _ = self.postprocessor.process_mask(mask, shape_type="rotated_rect")
        cropped = self.postprocessor.crop_image(
            image_bgr,
            processed_mask,
            padding=self.padding,
            background_color=(0, 0, 0)
        )

        # 保证输出为 RGB 格式和 [0,1] 范围
        roi_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        roi_rgb = roi_rgb.astype(np.float32) / 255.0

        return roi_rgb

    def extract_roi_with_crop_params(self, image_path: str) -> tuple:
        """
        从输入图像中提取ROI区域,并返回裁剪参数。

        参数:
            image_path: 输入图像路径
        返回:
            roi_rgb: numpy.ndarray, 形状(H,W,3), RGB顺序, 值范围[0,1]
            crop_params: dict, 包含裁剪参数 {'x', 'y', 'w', 'h', 'mask'}
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像不存在: {image_path}")

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"无法读取图像: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = self._predict_mask(image_rgb)

        processed_mask, _ = self.postprocessor.process_mask(mask, shape_type="rotated_rect")

        # 获取裁剪坐标
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            x, y, w, h = 0, 0, image_bgr.shape[1], image_bgr.shape[0]
        else:
            x, y, w, h = cv2.boundingRect(np.vstack(contours))

        # 添加padding
        x = max(0, x - self.padding)
        y = max(0, y - self.padding)
        w = min(image_bgr.shape[1] - x, w + 2 * self.padding)
        h = min(image_bgr.shape[0] - y, h + 2 * self.padding)

        cropped = self.postprocessor.crop_image(
            image_bgr,
            processed_mask,
            padding=self.padding,
            background_color=(0, 0, 0)
        )

        # 保证输出为 RGB 格式和 [0,1] 范围
        roi_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        roi_rgb = roi_rgb.astype(np.float32) / 255.0

        crop_params = {
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'mask': processed_mask
        }

        return roi_rgb, crop_params


# ========== 调试入口 ==========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="测试 ROI 提取器")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--output", default="./roi_test_output.png", help="输出ROI路径")
    args = parser.parse_args()

    extractor = ROIExtractor(args.checkpoint)
    roi = extractor.extract_roi(args.image)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    roi_bgr = (roi[:, :, ::-1] * 255).astype(np.uint8)
    cv2.imwrite(args.output, roi_bgr)
    print(f"ROI 已保存到: {args.output}")
