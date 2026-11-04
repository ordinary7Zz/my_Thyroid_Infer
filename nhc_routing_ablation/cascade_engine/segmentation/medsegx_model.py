"""
MedSegX 分割模型（SAM + MoE Adapter）。

架构：SAM 基础模型 + task-specific adapter（MoE）+ modal/organ embedding。
需要 box prompt（Agent 场景用全图 box）+ task_name。

L2 模型：依赖 segment_anything（本地修改版），通过 infer_root 引入 sys.path。
推理逻辑复制自 infer_medsegx/inference.py 的 infer_single 函数。
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


@register_seg_model("medsegx")
class MedSegXSegmentationModel(BaseSegmentationModel):
    """MedSegX (SAM + MoE Adapter) 分割模型。"""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        sam_checkpoint: str,
        sam_model_type: str = "vit_b",
        task_name: str = "US_ThyroidNodule",
        img_size: int = 256,
        threshold: float = 0.5,
        bottleneck_dim: int = 16,
        embedding_dim: int = 16,
        expert_num: int = 4,
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
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.task_name = task_name
        self.img_size = img_size
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.expert_num = expert_num
        self.infer_root = infer_root
        self._modal: Optional[int] = None
        self._organ: Optional[tuple] = None

    def load_model(self) -> None:
        print(f"  加载 MedSegX {self.model_name}")

        if self.infer_root:
            sys.path.insert(0, self.infer_root)

        try:
            from segment_anything import sam_model_registry
            from model.medsegx import MedSegX
        except ImportError as e:
            raise ImportError(
                f"无法导入 MedSegX 架构。请确认 infer_root 配置正确: {e}"
            )

        # SAM 权重路径：sam_checkpoint 可能是目录名或完整文件路径
        sam_model_checkpoint = {
            "vit_b": "sam_vit_b_01ec64.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_h": "sam_vit_h_4b8939.pth",
        }
        expected_filename = sam_model_checkpoint.get(
            self.sam_model_type, "sam_vit_b_01ec64.pth"
        )
        if os.path.isdir(self.sam_checkpoint):
            sam_ckpt_path = os.path.join(self.sam_checkpoint, expected_filename)
        else:
            sam_ckpt_path = self.sam_checkpoint

        sam_model = sam_model_registry[self.sam_model_type](
            image_size=self.img_size,
            keep_resolution=True,
            checkpoint=sam_ckpt_path,
        )

        self.model = MedSegX(
            sam=sam_model,
            bottleneck_dim=self.bottleneck_dim,
            embedding_dim=self.embedding_dim,
            expert_num=self.expert_num,
        )

        # 加载 MedSegX 微调权重
        ckpt = torch.load(self.model_path, map_location=self.device)
        self.model.load_parameters(ckpt["model"])

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

        # 解析 task_name → modal + organ
        try:
            from inference import parse_task
            self._modal, self._organ = parse_task(self.task_name)
        except (ImportError, Exception) as e:
            print(f"  ⚠️ parse_task 失败 ({e})，使用默认值 modal=0, organ=(0,0,0,0)")
            self._modal = 0
            self._organ = (0, 0, 0, 0)

        print(f"    ✓ MedSegX 加载成功 (task={self.task_name}, modal={self._modal})")

    def predict(self, image: np.ndarray) -> SegModelOutput:
        """
        分割推理（全图 box prompt）。

        Args:
            image: (H, W, 3) RGB uint8 [0,255].
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载")

        from torchvision.transforms.functional import resize as tv_resize

        h_orig, w_orig = image.shape[:2]

        # Image → tensor (C,H,W) → (1,3,img_size,img_size)
        img_tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        img_tensor = tv_resize(
            img_tensor.unsqueeze(0), [self.img_size, self.img_size], antialias=True
        )

        # 全图 box
        box = np.array([0, 0, w_orig, h_orig], dtype=np.float32)
        box = torch.tensor([box], dtype=torch.float32)

        # box 变换
        try:
            from segment_anything.transforms import ResizeLongestSide
            box_transform = ResizeLongestSide(self.img_size)
            box = box_transform.apply_boxes_torch(
                box.reshape(-1, 2, 2), (h_orig, w_orig)
            ).reshape(-1, 4)
        except ImportError:
            # 简化：直接缩放 box 到 img_size
            scale = self.img_size / max(h_orig, w_orig)
            box = box * scale

        img_tensor = img_tensor.to(self.device)
        box = box.to(self.device)
        img_tensor = self.model.sam.preprocess(img_tensor)

        # Prompt encoder
        sparse_emb, dense_emb = self.model.sam.prompt_encoder(
            points=None, boxes=box[:, None, :], masks=None
        )

        # Modal & organ embedding
        modal_t = torch.tensor([self._modal], dtype=torch.long, device=self.device)
        modal_index = self.model.sam.image_encoder.modal_index[modal_t]
        modal_embed = self.model.sam.image_encoder.modal_embed(modal_index)

        o1, o2, o3, o4 = self._organ
        o1_t = torch.tensor([o1], dtype=torch.long, device=self.device)
        o2_t = torch.tensor([o2], dtype=torch.long, device=self.device)
        o3_t = torch.tensor([o3], dtype=torch.long, device=self.device)
        o4_t = torch.tensor([o4], dtype=torch.long, device=self.device)
        organ_index_0 = torch.zeros(1, dtype=torch.long, device=self.device)
        organ_embed = (
            self.model.sam.image_encoder.organ_embed[0](organ_index_0),
            self.model.sam.image_encoder.organ_embed[1](
                self.model.sam.image_encoder.organ_index_1[o1_t]
            ),
            self.model.sam.image_encoder.organ_embed[2](
                self.model.sam.image_encoder.organ_index_2[o2_t]
            ),
            self.model.sam.image_encoder.organ_embed[3](
                self.model.sam.image_encoder.organ_index_3[o3_t]
            ),
            self.model.sam.image_encoder.organ_embed[4](o4_t),
        )

        # Image encoder
        with torch.no_grad():
            image_embedding, _ = self.model.sam.image_encoder(
                img_tensor, modal_embed, organ_embed
            )

            # Mask decoder
            mask_pred, iou_pred = self.model.sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )

        # 取最高 IoU 的 mask
        best_idx = iou_pred.argmax(dim=1)
        mask_prob = torch.sigmoid(mask_pred)
        chosen_prob = mask_prob[0, best_idx[0]].cpu().numpy().astype(np.float32)
        chosen_mask = (chosen_prob > self.threshold).astype(np.uint8)

        del mask_pred, iou_pred, image_embedding, img_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # resize 回原图
        if chosen_mask.shape != (h_orig, w_orig):
            mask_tensor = torch.from_numpy(chosen_mask).unsqueeze(0).unsqueeze(0).float()
            mask_resized = (
                F.interpolate(
                    mask_tensor, size=(h_orig, w_orig),
                    mode="bilinear", align_corners=False,
                )
                .squeeze()
                .numpy()
            )
            mask = (mask_resized > 0.5).astype(np.uint8)
            conf_map = cv2.resize(
                chosen_prob, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
        else:
            mask = chosen_mask
            conf_map = chosen_prob

        return SegModelOutput(
            model_name=self.model_name,
            mask=mask,
            confidence_map=conf_map,
            metadata=self.get_metadata(),
        )
