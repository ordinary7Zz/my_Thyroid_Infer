"""
DINO-UNet 分割模型包装
基于 DINOv3 + UNet 的分割模型，用于甲状腺结节分割
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

from .base_model import BaseSegmentationModel, ModelOutput
from model_architectures.dino_unet import DINOv3_S_UNet


class DINOUNetSegmentationModel(BaseSegmentationModel):
    """
    DINO-UNet 分割模型
    - 输入：图像（RGB）
    - 输出：分割掩码（二值）
    """
    
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        training_data_devices: list = None,
        input_size: tuple = (448, 448),
        threshold: float = 0.5,
        use_dilation: bool = None,  # None 表示自动检测
        **kwargs
    ):
        """
        初始化 DINO-UNet 分割模型
        
        Args:
            model_name: 模型名称
            model_path: 模型权重路径
            device: 运行设备
            training_data_devices: 训练数据设备列表
            input_size: 输入图像尺寸 (height, width)
            threshold: 二值化阈值
            use_dilation: 是否使用 dilation（None 表示从权重文件自动检测）
            **kwargs: 其他参数
        """
        # 先设置这些属性，因为 super().__init__() 会调用 load_model()
        # 而 load_model() 需要访问 self.use_dilation
        self.input_size = input_size
        self.threshold = threshold
        self.use_dilation = use_dilation
        
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            device=device,
            training_data_devices=training_data_devices,
            input_size=input_size,
            threshold=threshold,
            use_dilation=use_dilation,
            **kwargs
        )
        
        # 统一图像预处理方式
        # 读取 input_size 的 height 和 width（不要求必须一致）
        if isinstance(self.input_size, (tuple, list)):
            resize_size = (self.input_size[0], self.input_size[1])  # (height, width)
        else:
            # 如果是单个值，则使用正方形
            resize_size = (self.input_size, self.input_size)
        
        self.transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self) -> None:
        """加载 DINO-UNet 模型"""
        print(f"  加载 {self.model_name} 从 {self.model_path}")
        
        if not self.model_path:
            raise ValueError("Model path is required")
        
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
        try:
            # 自动检测 use_dilation 参数
            use_dilation = self.use_dilation
            state_dict = None
            
            if use_dilation is None:
                # 从权重文件中检测
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 处理不同的 checkpoint 格式
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # 检查 state_dict 中是否包含 dilation 相关的键
                if state_dict is not None:
                    dilation_keys = [key for key in state_dict.keys() if 'dilate' in key]
                    use_dilation = len(dilation_keys) > 0
                    if use_dilation:
                        print(f"    检测到 dilation 参数，使用 use_dilation=True")
                    else:
                        print(f"    未检测到 dilation 参数，使用 use_dilation=False")
            else:
                # 如果指定了 use_dilation，也需要加载权重来检查
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
            
            # 创建模型
            self.model = DINOv3_S_UNet(pretrained=False, use_dilation=use_dilation)
            
            # 加载权重
            if state_dict is not None:
                try:
                    self.model.load_state_dict(state_dict, strict=True)
                    print(f"    ✓ 成功加载权重（严格模式）")
                except RuntimeError as e:
                    # 如果严格模式失败，尝试非严格模式
                    if "Unexpected key(s)" in str(e) or "Missing key(s)" in str(e):
                        print(f"    警告: 权重键不完全匹配，使用非严格模式加载")
                        # 获取模型当前 state_dict 的键
                        model_keys = set(self.model.state_dict().keys())
                        state_dict_keys = set(state_dict.keys())
                        
                        # 只加载匹配的键
                        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
                        missing_keys = model_keys - state_dict_keys
                        unexpected_keys = state_dict_keys - model_keys
                        
                        if missing_keys:
                            print(f"    缺失的键: {len(missing_keys)} 个（将使用随机初始化）")
                        if unexpected_keys:
                            print(f"    额外的键: {len(unexpected_keys)} 个（将被忽略）")
                        
                        self.model.load_state_dict(filtered_state_dict, strict=False)
                        print(f"    ✓ 成功加载权重（非严格模式）")
                    else:
                        raise
            else:
                print(f"    ⚠️  警告: 无法读取权重文件，使用随机初始化")
            
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            
        except Exception as e:
            print(f"    ✗ 加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: 输入图像 (H, W, 3) numpy array, RGB格式, float32, [0, 1] 范围
            
        Returns:
            处理后的张量 (1, 3, H, W)
        """
        # 确保图像在 [0, 1] 范围，转换为 uint8 [0, 255]
        if image.dtype == np.float32 or image.dtype == np.float64:
            # 如果值域超出 [0, 1]，先裁剪
            if np.min(image) < 0 or np.max(image) > 1:
                image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
        
        # 转换为 PIL Image
        pil_image = Image.fromarray(image)
        
        # 应用统一的预处理变换（Resize + ToTensor + Normalize）
        tensor = self.transform(pil_image)
        
        # 添加批次维度: (3, H, W) -> (1, 3, H, W)
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, image: np.ndarray) -> ModelOutput:
        """
        运行分割推理
        
        Args:
            image: 输入图像 (H, W, 3) numpy array, RGB格式, float32, [0, 1]
            
        Returns:
            ModelOutput 包含分割掩码
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # 保存原始尺寸
        original_shape = image.shape[:2]
        
        # 预处理
        input_tensor = self.preprocess(image)
        
        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)  # (1, 1, H, W)
            # 立即转换到 CPU 并转为 numpy，同时删除 GPU 张量
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()  # (H, W)
            
            # 删除 GPU 上的张量，释放显存
            del output, input_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 后处理得到二值掩码
        mask = self.postprocess(prob_map, original_shape, self.threshold)
        
        # Resize 概率图到原始尺寸（用于置信度评估）
        if prob_map.shape != original_shape:
            prob_map_resized = cv2.resize(
                prob_map, 
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            prob_map_resized = prob_map
        
        # 创建 ModelOutput
        return ModelOutput(
            model_name=self.model_name,
            mask=mask,
            confidence_map=prob_map_resized,
            metadata=self.get_metadata()
        )
    
    def postprocess(
        self, 
        output: np.ndarray, 
        original_shape: tuple,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        后处理模型输出为二值掩码
        
        Args:
            output: 概率图 (H, W)
            original_shape: 原始图像尺寸 (H, W)
            threshold: 二值化阈值
            
        Returns:
            二值掩码 (H, W) with values 0 or 1
        """
        # 确保 2D
        while output.ndim > 2:
            output = output.squeeze(0)
        
        # Resize 到原始尺寸
        if output.shape != original_shape:
            output = cv2.resize(
                output, 
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        # 阈值化
        mask = (output > threshold).astype(np.uint8)
        
        # 可选：形态学后处理（去除小连通区域）
        mask = self._postprocess_morphology(mask)
        
        return mask
    
    def _postprocess_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        应用形态学操作清理掩码
        
        Args:
            mask: 二值掩码 (H, W)
            
        Returns:
            清理后的二值掩码
        """
        # 去除小连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        # 保留面积大于阈值的连通区域
        min_area = 50  # 像素
        cleaned_mask = np.zeros_like(mask)
        
        for i in range(1, num_labels):  # 跳过背景 (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned_mask[labels == i] = 1
        
        # 可选：形态学闭运算填充小孔
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        return cleaned_mask

