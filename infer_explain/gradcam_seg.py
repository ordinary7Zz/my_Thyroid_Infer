# 本脚本支持对单张图像或目录中的多张图像在 DINOv3_S_UNet 分割网络中使用 Grad-CAM 生成热力图，
# 输出风格与 heatmap.jpg 一致：大面积扩散热力 + 原图清晰透出 + jet colormap + 固定 alpha 叠加。

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F
from torchvision import transforms

if TYPE_CHECKING:
    from model import DINOv3_S_UNet


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OUTPUT_TYPE_CHOICES = (
    "all",
    "original",
    "overlay",
    "overlay_gt",
    "original_gt",
    "gradcam_map",
)
OUTPUT_SUBDIRS = {
    "original": "original",
    "overlay": "overlay",
    "overlay_gt": "overlay_gt",
    "original_gt": "original_gt",
    "gradcam_map": "gradcam_map",
}


def collect_files_by_stem(directory: Path, suffixes: set[str]) -> Dict[str, Path]:
    """收集目录中支持后缀的文件，并按 stem 建立映射。"""
    if not directory.is_dir():
        raise NotADirectoryError(f"目录不存在或不是目录: {directory}")

    files: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        stem = path.stem
        if stem in files:
            raise ValueError(f"发现重复文件名（去掉后缀后同名）：{stem}")
        files[stem] = path
    return files


def resolve_output_types(output_type: str) -> Tuple[str, ...]:
    """解析输出类型参数。"""
    normalized = output_type.lower()
    if normalized not in OUTPUT_TYPE_CHOICES:
        raise ValueError(
            f"不支持的 output_type: {output_type}，可选值为: {', '.join(OUTPUT_TYPE_CHOICES)}"
        )
    if normalized == "all":
        return ("original", "overlay", "overlay_gt", "original_gt", "gradcam_map")
    return (normalized,)


def resolve_input_pairs(
    image_path: Optional[str],
    mask_path: Optional[str],
    image_dir: Optional[str],
    mask_dir: Optional[str],
    output_types: Sequence[str],
) -> list[tuple[str, Optional[str]]]:
    """根据单图或目录模式解析待处理的图像与掩码配对列表。"""
    use_single_image = image_path is not None
    use_image_dir = image_dir is not None

    if use_single_image == use_image_dir:
        raise ValueError("必须且只能提供一个输入来源：image_path 或 image_dir。")

    requires_mask = any(output_type in {"overlay_gt", "original_gt"} for output_type in output_types)

    if use_single_image:
        if mask_dir is not None:
            raise ValueError("单图模式下不能同时提供 mask_dir。")

        image_file = Path(image_path)
        if not image_file.is_file():
            raise FileNotFoundError(f"图像文件不存在: {image_file}")

        resolved_mask_path: Optional[str] = None
        if mask_path is not None:
            mask_file = Path(mask_path)
            if not mask_file.is_file():
                raise FileNotFoundError(f"掩码文件不存在: {mask_file}")
            resolved_mask_path = str(mask_file)

        if requires_mask and resolved_mask_path is None:
            raise ValueError("当前输出类型需要提供 mask_path 或 mask_dir。")

        return [(str(image_file), resolved_mask_path)]

    if mask_path is not None:
        raise ValueError("目录模式下不能同时提供 mask_path，请改用 mask_dir。")

    image_files = collect_files_by_stem(Path(image_dir), IMAGE_EXTENSIONS)
    if not image_files:
        raise ValueError(f"图像目录中未找到支持的图像文件: {image_dir}")

    mask_files: Dict[str, Path] = {}
    if mask_dir is not None:
        mask_files = collect_files_by_stem(Path(mask_dir), IMAGE_EXTENSIONS)

    if requires_mask and not mask_files:
        raise ValueError("当前输出类型需要提供 mask_dir，并确保每张图像都有对应掩码。")

    resolved_pairs: list[tuple[str, Optional[str]]] = []
    missing_mask_stems: list[str] = []

    for stem, image_file in image_files.items():
        mask_file = mask_files.get(stem)
        if mask_dir is not None and mask_file is None:
            missing_mask_stems.append(stem)
            continue
        resolved_pairs.append((str(image_file), str(mask_file) if mask_file is not None else None))

    if missing_mask_stems:
        preview = ", ".join(missing_mask_stems[:10])
        if len(missing_mask_stems) > 10:
            preview += " ..."
        raise ValueError(f"以下图像缺少对应 mask（按 stem 匹配）: {preview}")

    return resolved_pairs


def build_output_paths(
    output_dir: str,
    image_filename: str,
    output_types: Sequence[str],
) -> Dict[str, str]:
    """为当前图像构建输出文件路径，并按需创建子目录。"""
    output_root = Path(output_dir)
    output_paths: Dict[str, str] = {}
    for output_type in output_types:
        subdir = output_root / OUTPUT_SUBDIRS[output_type]
        subdir.mkdir(parents=True, exist_ok=True)
        output_paths[output_type] = str(subdir / image_filename)
    return output_paths


def load_image(
    image_path: str,
    img_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    """加载单张图像并做预处理。"""
    img = Image.open(image_path).convert("RGB")
    resized_img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    orig_np = np.array(resized_img)  # (H, W, 3) uint8

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    img_t = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)
    return img_t, orig_np


def load_region_mask(mask_path: Optional[str], img_size: int) -> Optional[np.ndarray]:
    """加载并二值化区域掩码。"""
    if mask_path is None:
        return None

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((img_size, img_size), resample=Image.NEAREST)
    mask_np = np.array(mask)
    return (mask_np > mask_np.mean()).astype(np.float32)


def build_model(
    checkpoint: str,
    device: torch.device,
    dino_pretrained: bool = True,
) -> "DINOv3_S_UNet":
    """构建 DINOv3_S_UNet 并加载 checkpoint 权重。"""
    from model import DINOv3_S_UNet

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = DINOv3_S_UNet(pretrained=dino_pretrained).to(device)
    model.eval()

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    return model


class GradCAM:
    """
    Grad-CAM for segmentation model.

    通过 hook 获取指定层的特征图和梯度，
    计算加权特征图作为 CAM 热力图。
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册前向和反向 hook
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        生成 Grad-CAM 热力图。

        Args:
            input_tensor: (1, 3, H, W) 输入张量
            target_mask: 可选的目标区域掩膜 (H, W)，用于指定计算梯度的目标区域

        Returns:
            cam: (H, W) 归一化到 [0, 1] 的热力图
        """
        self.model.zero_grad()

        # 前向传播
        output = self.model(input_tensor)  # (1, 1, H, W)

        if target_mask is not None:
            mask_tensor = torch.from_numpy(target_mask).float().to(input_tensor.device)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            mask_tensor = F.interpolate(mask_tensor, size=output.shape[2:], mode="nearest")
            target_score = (output * mask_tensor).sum() / (mask_tensor.sum() + 1e-6)
        else:
            flat_logits = output.flatten()
            k = max(1, flat_logits.numel() // 20)
            target_score = flat_logits.topk(k).values.mean()

        # 反向传播获取梯度
        target_score.backward()

        # 获取梯度和激活
        gradients = self.gradients  # (1, C, h, w)
        activations = self.activations  # (1, C, h, w)

        # Global Average Pooling on gradients -> channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # 加权求和
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # ReLU: 只保留正向贡献
        cam = F.relu(cam)

        # 上采样到输入图像尺寸
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )

        # 转为 numpy
        cam = cam.squeeze().cpu().numpy()  # (H, W)

        return cam

    def release(self):
        self._forward_hook.remove()
        self._backward_hook.remove()


def postprocess_cam(
    cam: np.ndarray,
    smooth_sigma_ratio: float = 0.02,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    对原始 CAM 进行后处理，使热力分布接近 heatmap.jpg 风格。

    处理步骤：
    1. percentile 裁切归一化（避免极端值拉伸）
    2. 大核高斯平滑（让热力扩散、过渡宽）
    3. 重新归一化
    4. gamma < 1 变换（抬升中低值，扩大绿/黄过渡区域面积）

    Args:
        cam: (H, W) 原始 Grad-CAM 热力图
        smooth_sigma_ratio: 高斯平滑核大小占图像尺寸的比例，越大越扩散
        gamma: gamma 变换指数，< 1 抬升中低值，让热力分布更"满"

    Returns:
        cam_final: (H, W) [0, 1] 后处理后的热力图
    """
    H, W = cam.shape

    # 1) percentile 裁切归一化：截断极端值，保留主体分布
    p_low, p_high = np.percentile(cam, [1, 99])
    cam = np.clip(cam, p_low, p_high)
    c_min, c_max = cam.min(), cam.max()
    if c_max - c_min > 1e-8:
        cam = (cam - c_min) / (c_max - c_min)
    else:
        cam = np.zeros_like(cam)

    # 2) 轻量高斯平滑：仅用于展示，不强行塑形
    sigma = max(H, W) * smooth_sigma_ratio
    if sigma > 1e-6:
        cam = gaussian_filter(cam, sigma=sigma)

    # 3) 重新归一化
    c_min, c_max = cam.min(), cam.max()
    if c_max - c_min > 1e-8:
        cam = (cam - c_min) / (c_max - c_min)
    else:
        cam = np.zeros_like(cam)

    # 4) gamma 变换：默认不压缩/抬升中低值，尽量保留原始响应形状
    cam = np.power(cam, gamma)

    return cam


def create_heatmap_rgb(cam: np.ndarray) -> np.ndarray:
    """将归一化 CAM 转换为彩色热力图。"""
    cam_uint8 = np.uint8(255 * cam)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def generate_heatmap_overlay(
    cam: np.ndarray,
    orig_np: np.ndarray,
    alpha: float = 0.65,
    saturation_scale: float = 1.3,
) -> np.ndarray:
    """
    生成与 heatmap.jpg 风格一致的叠加图像。

    关键风格特征：
    - jet colormap
    - 固定 alpha 叠加（整幅图均匀覆盖）
    - 原图纹理清晰透出
    - 色彩饱和鲜艳（HSV 饱和度增强）

    Args:
        cam: (H, W) 归一化的热力图 [0, 1]
        orig_np: (H, W, 3) uint8 原图
        alpha: 热力图叠加的不透明度（0.65 更接近 heatmap.jpg）
        saturation_scale: 饱和度增强倍数（>1 更鲜艳）

    Returns:
        blended: (H, W, 3) uint8 叠加图像
    """
    H, W = orig_np.shape[:2]

    # 确保 cam 尺寸匹配
    if cam.shape != (H, W):
        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)

    heatmap_rgb = create_heatmap_rgb(cam)

    # 固定 alpha 叠加：blended = alpha * heatmap + (1 - alpha) * orig
    orig_float = orig_np.astype(np.float32)
    heatmap_float = heatmap_rgb.astype(np.float32)

    blended = alpha * heatmap_float + (1.0 - alpha) * orig_float
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # 色彩饱和度增强：在 HSV 空间提升 S 通道，让颜色更鲜艳
    if saturation_scale != 1.0:
        blended_hsv = cv2.cvtColor(blended, cv2.COLOR_RGB2HSV).astype(np.float32)
        blended_hsv[:, :, 1] = np.clip(blended_hsv[:, :, 1] * saturation_scale, 0, 255)
        blended = cv2.cvtColor(blended_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return blended


def resolve_target_layer(
    model: torch.nn.Module,
    target_layer_name: str,
) -> tuple[torch.nn.Module, str]:
    """根据名称选择 Grad-CAM 目标层。"""
    target_layer = getattr(model, target_layer_name, None)
    resolved_target_layer_name = target_layer_name
    if target_layer is None:
        resolved_target_layer_name = "reduce4"
        print(f"Warning: target_layer '{target_layer_name}' not found, using 'reduce4'")
        target_layer = model.reduce4
    return target_layer, resolved_target_layer_name


def draw_mask_contours_on_image(
    base_image: np.ndarray,
    region_mask: Optional[np.ndarray],
) -> np.ndarray:
    """在底图上绘制 GT mask 轮廓。"""
    if region_mask is None:
        raise ValueError("GT 叠加输出需要提供对应的 mask。")

    contours_img = base_image.copy()
    mask_uint8 = (region_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 2)
    return contours_img


def save_sample_outputs(
    image_path: str,
    output_dir: str,
    output_types: Sequence[str],
    orig_np: np.ndarray,
    cam: np.ndarray,
    region_mask: Optional[np.ndarray],
    alpha: float,
    saturation_scale: float,
) -> Dict[str, str]:
    """按输出类型保存当前样本的可视化结果。"""
    image_filename = Path(image_path).name
    output_paths = build_output_paths(output_dir, image_filename, output_types)
    saved_paths: Dict[str, str] = {}

    heatmap_rgb: Optional[np.ndarray] = None
    blended: Optional[np.ndarray] = None

    if "gradcam_map" in output_paths:
        heatmap_rgb = create_heatmap_rgb(cam)
        Image.fromarray(heatmap_rgb).save(output_paths["gradcam_map"])
        saved_paths["gradcam_map"] = output_paths["gradcam_map"]

    if any(output_type in output_paths for output_type in ("overlay", "overlay_gt")):
        blended = generate_heatmap_overlay(
            cam,
            orig_np,
            alpha=alpha,
            saturation_scale=saturation_scale,
        )

    if "original" in output_paths:
        Image.fromarray(orig_np).save(output_paths["original"])
        saved_paths["original"] = output_paths["original"]

    if "overlay" in output_paths:
        if blended is None:
            raise RuntimeError("overlay 输出生成失败。")
        Image.fromarray(blended).save(output_paths["overlay"])
        saved_paths["overlay"] = output_paths["overlay"]

    if "overlay_gt" in output_paths:
        if blended is None:
            raise RuntimeError("overlay_gt 输出生成失败。")
        contours_img = draw_mask_contours_on_image(blended, region_mask)
        Image.fromarray(contours_img).save(output_paths["overlay_gt"])
        saved_paths["overlay_gt"] = output_paths["overlay_gt"]

    if "original_gt" in output_paths:
        original_gt_img = draw_mask_contours_on_image(orig_np, region_mask)
        Image.fromarray(original_gt_img).save(output_paths["original_gt"])
        saved_paths["original_gt"] = output_paths["original_gt"]

    return saved_paths


def run_gradcam(
    checkpoint: str,
    output_dir: str,
    image_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    image_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
    output_type: str = "all",
    img_size: int = 224,
    dino_pretrained: bool = True,
    alpha: float = 0.45,
    target_layer_name: str = "reduce4",
    smooth_sigma_ratio: float = 0.02,
    gamma: float = 1.0,
    saturation_scale: float = 1.3,
) -> None:
    """
    对单张图像或目录中的图像生成 Grad-CAM 热力图。

    输出规则：
    - 同一类输出保存到对应子目录下
    - 文件名与原图保持一致
    - 可选输出目录: original / overlay / overlay_gt / original_gt / gradcam_map
    """
    output_types = resolve_output_types(output_type)
    input_pairs = resolve_input_pairs(
        image_path=image_path,
        mask_path=mask_path,
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_types=output_types,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Resolved output types: {', '.join(output_types)}")
    print(f"Number of images to process: {len(input_pairs)}")

    model = build_model(checkpoint, device, dino_pretrained=dino_pretrained)
    model.eval()

    target_layer, resolved_target_layer_name = resolve_target_layer(model, target_layer_name)
    print(f"Using target layer: {resolved_target_layer_name}")

    gradcam = GradCAM(model, target_layer)
    try:
        for index, (current_image_path, current_mask_path) in enumerate(input_pairs, start=1):
            print(f"[{index}/{len(input_pairs)}] Processing image: {current_image_path}")

            img_tensor, orig_np = load_image(current_image_path, img_size, device)
            print(f"Loaded image: {current_image_path} -> ({img_size}, {img_size})")

            region_mask = load_region_mask(current_mask_path, img_size)
            if current_mask_path is not None:
                print(f"Loaded mask: {current_mask_path}")

            img_tensor.requires_grad_(True)
            cam_raw = gradcam.generate(img_tensor, target_mask=region_mask)
            print(
                f"Grad-CAM raw computed, shape: {cam_raw.shape}, "
                f"range: [{cam_raw.min():.4f}, {cam_raw.max():.4f}]"
            )

            cam = postprocess_cam(
                cam_raw,
                smooth_sigma_ratio=smooth_sigma_ratio,
                gamma=gamma,
            )
            print(
                f"Postprocessed CAM, range: [{cam.min():.4f}, {cam.max():.4f}], "
                f"sigma_ratio={smooth_sigma_ratio}, gamma={gamma}"
            )

            saved_paths = save_sample_outputs(
                image_path=current_image_path,
                output_dir=output_dir,
                output_types=output_types,
                orig_np=orig_np,
                cam=cam,
                region_mask=region_mask,
                alpha=alpha,
                saturation_scale=saturation_scale,
            )
            for saved_type, saved_path in saved_paths.items():
                print(f"Saved {saved_type}: {saved_path}")
    finally:
        gradcam.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对单张图像或目录中的 DINOv3_S_UNet 分割结果生成 Grad-CAM 热力图（heatmap.jpg 风格）。"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="dino_unet 的权重文件路径（.pth 或 .pt）。",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="待分析的单张图像路径；与 image_dir 二选一。",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="单图模式下可选：对应的 mask 路径，用作 Grad-CAM 目标区域。",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="批量模式下的图像目录；与 image_path 二选一。",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="批量模式下的 mask 目录，按文件名 stem 与 image_dir 中图像配对。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gradcam_single_image_out",
        help="保存 Grad-CAM 可视化结果的根目录。",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="all",
        choices=OUTPUT_TYPE_CHOICES,
        help="输出类型：all/original/overlay/overlay_gt/original_gt/gradcam_map。",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="输入图像缩放尺寸，需与训练/测试时保持一致。",
    )
    parser.add_argument(
        "--dino_pretrained",
        type=str,
        default="True",
        help="是否为 DINOv3 backbone 加载 ImageNet 预训练权重（True/False）。",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="热力图叠加不透明度（0~1）。0.45 原图纹理清晰透出。",
    )
    parser.add_argument(
        "--target_layer",
        type=str,
        default="reduce4",
        choices=["up1", "up2", "up3", "up4", "reduce1", "reduce2", "reduce3", "reduce4"],
        help="Grad-CAM 目标层名称。reduce4（默认）更偏语义关注，避免过于贴近输出。",
    )
    parser.add_argument(
        "--smooth_sigma_ratio",
        type=float,
        default=0.02,
        help="高斯平滑核占图像尺寸的比例；值越小越接近原始 CAM。",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma 变换指数；1.0 表示尽量保持原始响应形状。",
    )
    parser.add_argument(
        "--saturation_scale",
        type=float,
        default=1.3,
        help="色彩饱和度增强倍数（>1 更鲜艳）。",
    )
    args = parser.parse_args()
    args.dino_pretrained = str(args.dino_pretrained).lower() in ("true", "1", "yes", "y")
    return args


if __name__ == "__main__":
    args = parse_args()
    run_gradcam(
        checkpoint=args.checkpoint,
        image_path=args.image_path,
        mask_path=args.mask_path,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        output_type=args.output_type,
        img_size=args.img_size,
        dino_pretrained=args.dino_pretrained,
        alpha=args.alpha,
        target_layer_name=args.target_layer,
        smooth_sigma_ratio=args.smooth_sigma_ratio,
        gamma=args.gamma,
        saturation_scale=args.saturation_scale,
    )
