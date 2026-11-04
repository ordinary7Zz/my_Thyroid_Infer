"""
TransUNet 分割模型。

架构：ResNetV2 + ViT-B/16 混合编码器 + UNet-style 解码器。
输入灰度图（模型内复制为 3 通道），输出多类 logits。

L1 模型：依赖 scipy + ml_collections，架构文件在 infer_transunet/networks/ 下，
通过 infer_root 参数引入 sys.path。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base_model import BaseSegmentationModel, SegModelOutput
from .model_factory import register_seg_model


@register_seg_model("transunet")
class TransUNetSegmentationModel(BaseSegmentationModel):
    """TransUNet (R50-ViT-B_16) 分割模型。"""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        num_classes: int = 2,
        vit_name: str = "R50-ViT-B_16",
        img_size: int = 224,
        n_skip: int = 3,
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
        self.num_classes = num_classes
        self.vit_name = vit_name
        self.img_size = img_size
        self.n_skip = n_skip
        self.infer_root = infer_root

    def load_model(self) -> None:
        print(f"  加载 TransUNet {self.model_name} 从 {self.model_path}")

        # 引入架构定义
        if self.infer_root:
            sys.path.insert(0, self.infer_root)

        try:
            from networks.vit_seg_modeling import VisionTransformer as ViT_seg
        except ImportError as e:
            raise ImportError(
                f"无法导入 TransUNet 架构 (vit_seg_modeling)。请确认 infer_root 配置正确: {e}"
            )

        # 尝试多种导入方式，兼容不同版本的 TransUNet 代码
        CONFIGS_ViT_seg = None
        for import_expr in [
            ("networks.vit_seg_configs", "CONFIGS"),
            ("networks.vit_seg_configs", "configs"),
        ]:
            try:
                module_name, attr_name = import_expr
                module = __import__(module_name, fromlist=[attr_name])
                CONFIGS_ViT_seg = getattr(module, attr_name)
                break
            except (ImportError, AttributeError):
                continue

        if CONFIGS_ViT_seg is None:
            # vit_seg_configs.py 没有导出统一的 CONFIGS 或 get_config，
            # 而是定义了独立的 get_{xxx}_config() 函数。我们在此自行构建映射。
            vit_name_to_func = {
                "R50-ViT-B_16": "get_r50_b16_config",
                "R50-ViT-L_16": "get_r50_l16_config",
                "ViT-B_16": "get_b16_config",
                "ViT-B_32": "get_b32_config",
                "ViT-L_16": "get_l16_config",
                "ViT-L_32": "get_l32_config",
                "ViT-H_14": "get_h14_config",
                "testing": "get_testing",
            }
            func_name = vit_name_to_func.get(self.vit_name)
            if func_name is None:
                raise KeyError(
                    f"未知的 vit_name: '{self.vit_name}'，"
                    f"可选值: {list(vit_name_to_func.keys())}"
                )
            try:
                get_func = getattr(
                    __import__("networks.vit_seg_configs", fromlist=[func_name]),
                    func_name,
                )
                CONFIGS_ViT_seg = {self.vit_name: get_func()}
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"无法导入 TransUNet 配置函数 {func_name}: {e}"
                )

        config_vit = CONFIGS_ViT_seg[self.vit_name]
        config_vit.n_classes = self.num_classes
        config_vit.n_skip = self.n_skip
        if "R50" in self.vit_name:
            config_vit.patches.grid = (self.img_size // 16, self.img_size // 16)

        self.model = ViT_seg(
            config_vit,
            img_size=self.img_size,
            num_classes=self.num_classes,
        )

        # 加载权重（兼容三种格式）
        state = torch.load(self.model_path, map_location=self.device)
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
        self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        print(f"    ✓ TransUNet 加载成功")

    def predict(self, image: np.ndarray) -> SegModelOutput:
        """
        分割推理。

        Args:
            image: (H, W, 3) RGB uint8 [0,255]。内部转灰度。
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        from scipy.ndimage import zoom as scipy_zoom

        original_h, original_w = image.shape[:2]

        # RGB → 灰度
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # resize 到 img_size（三次样条插值）
        if gray.shape[0] != self.img_size or gray.shape[1] != self.img_size:
            gray_resized = scipy_zoom(
                gray,
                (self.img_size / gray.shape[0], self.img_size / gray.shape[1]),
                order=3,
            )
        else:
            gray_resized = gray

        # (1, 1, H, W) tensor，值 [0, 255]，无归一化
        input_tensor = (
            torch.from_numpy(gray_resized)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        with torch.no_grad():
            logits = self.model(input_tensor)  # (1, num_classes, H, W)
            probs = torch.softmax(logits, dim=1)  # (1, num_classes, H, W)

        # argmax 得到预测
        pred_resized = (
            torch.argmax(probs, dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        # 前景类概率作为 confidence_map
        if self.num_classes >= 2:
            conf_resized = probs[0, 1].cpu().numpy().astype(np.float32)
        else:
            conf_resized = probs[0, 0].cpu().numpy().astype(np.float32)

        del logits, probs, input_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # resize 回原图
        if pred_resized.shape != (original_h, original_w):
            pred = scipy_zoom(
                pred_resized,
                (original_h / pred_resized.shape[0], original_w / pred_resized.shape[1]),
                order=0,  # 最近邻
            )
            pred = np.round(pred).astype(np.uint8)
            conf_map = cv2.resize(
                conf_resized, (original_w, original_h), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
        else:
            pred = pred_resized
            conf_map = conf_resized

        mask = (pred > 0).astype(np.uint8)

        return SegModelOutput(
            model_name=self.model_name,
            mask=mask,
            confidence_map=conf_map,
            metadata=self.get_metadata(),
        )
