"""
MedSAM2 分割模型（SAM2 Video Predictor 单帧推理）。

架构：SAM2 video predictor，将单张图像当作"单帧视频"推理。
需要 box prompt（Agent 场景用全图 box）。

L2 模型：依赖 sam2（本地包），通过 infer_root 引入 sys.path。
推理逻辑复制自 infer_medsam2/infer.py 的 infer_one_image 函数。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base_model import BaseSegmentationModel, SegModelOutput
from .model_factory import register_seg_model


@register_seg_model("medsam2")
class MedSAM2SegmentationModel(BaseSegmentationModel):
    """MedSAM2 (SAM2 Video Predictor) 分割模型。"""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        model_cfg: str = "sam2_hiera_b+.yaml",
        img_size: int = 1024,
        threshold: float = 0.5,
        base_dataset_performance: Optional[dict] = None,
        dataset_info: Optional[dict] = None,
        device: str = "cuda",
        infer_root: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            input_size=(img_size, img_size),
            threshold=threshold,
            base_dataset_performance=base_dataset_performance,
            dataset_info=dataset_info,
            device=device,
            **kwargs,
        )
        self.model_cfg = model_cfg
        self.img_size = img_size
        self.infer_root = infer_root

    def load_model(self) -> None:
        print(f"  加载 MedSAM2 {self.model_name} 从 {self.model_path}")

        if self.infer_root:
            sys.path.insert(0, self.infer_root)

        try:
            from sam2.build_sam import build_sam2_video_predictor_npz
        except ImportError as e:
            raise ImportError(
                f"无法导入 sam2。请确认 infer_root 配置正确: {e}"
            )

        # 查找配置文件
        cfg_path = self._find_config_path()

        self.model = build_sam2_video_predictor_npz(
            config_file=cfg_path,
            ckpt_path=self.model_path,
            device=self.device,
        )

        self.model.eval()
        self.is_loaded = True
        print(f"    ✓ MedSAM2 加载成功")

    def _find_config_path(self) -> str:
        """查找 SAM2 配置文件。

        build_sam2 内部使用 hydra.compose(config_name=...)，config_name 必须为
        Hydra 搜索路径（pkg://sam2）下的相对路径，而非 file:// URI 或文件系统绝对路径。
        """
        if self.infer_root:
            # 验证文件存在于磁盘上（仅诊断用，不改变返回值格式）
            disk_path = os.path.join(self.infer_root, "sam2", "configs", self.model_cfg)
            if not os.path.isfile(disk_path):
                disk_path = os.path.join(self.infer_root, self.model_cfg)
            if not os.path.isfile(disk_path):
                raise FileNotFoundError(
                    f"SAM2 配置文件未找到: {self.model_cfg} "
                    f"(搜索: {self.infer_root})"
                )
        return f"configs/{self.model_cfg}"

    # 与 infer_medsam2/infer.py 保持一致
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def predict(self, image: np.ndarray) -> SegModelOutput:
        """
        分割推理（单帧当"视频"，全图 box prompt）。

        Args:
            image: (H, W, 3) RGB uint8 [0,255].
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        h_orig, w_orig = image.shape[:2]
        image_size = self.model.image_size

        # 预处理：(H,W,3) uint8 → (3,H,W) float [0,1] → resize → ImageNet normalize
        img_float = image.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1)  # (3,H,W)

        # resize 到 image_size
        img_resized = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # (3, image_size, image_size)

        # ImageNet 归一化（SAM2 模型期望）
        img_resized = (img_resized - self.IMAGENET_MEAN) / self.IMAGENET_STD

        images = img_resized.unsqueeze(0).to(self.device)  # (1, 3, image_size, image_size)

        # 单帧"视频"推理
        with torch.no_grad():
            inference_state = self.model.init_state(
                images=images,
                video_height=h_orig,
                video_width=w_orig,
            )

            # 全图 box
            box = np.array([0, 0, w_orig - 1, h_orig - 1], dtype=np.float32)
            _, _, out_mask_logits = self.model.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box=box,
            )

            # out_mask_logits: [num_obj, 1, H_orig, W_orig]
            mask_logits = out_mask_logits[0].float()  # [1, H, W]

            self.model.reset_state(inference_state)

        # sigmoid → 概率
        mask_prob = torch.sigmoid(mask_logits).squeeze().cpu().numpy().astype(np.float32)
        mask = (mask_prob > self.threshold).astype(np.uint8)

        del mask_logits, out_mask_logits, images, img_resized
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return SegModelOutput(
            model_name=self.model_name,
            mask=mask,
            confidence_map=mask_prob,
            metadata=self.get_metadata(),
        )
