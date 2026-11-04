"""
DINOv3_S_UNet_MULTITASK — 分割 + 分类多任务网络。

分割部分与 DINOv3_S_UNet 结构一致，但 DoubleConv 使用 GroupNorm
（而非 BatchNorm2d），权重与纯分割版不兼容。

分类部分：
  - 对 DINOv3 所有特征层做 GAP + GMP
  - 层注意力（3 层 → softmax 权重）融合多层特征
  - 特征注意力（sigmoid 门控）
  - 分类 backbone: 768 → 512 → 256
  - 两头：benign_malignant (1), tirads (5)

Forward 返回: (seg_out, benign_malignant, tirads)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DilatedConvBlock(nn.Module):
    """三路并行膨胀卷积（dilation 1/2/4）+ 1x1 融合。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4)
        self.fuse = nn.Conv2d(out_channels * 3, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return self.act(self.bn(self.fuse(x)))


class DoubleConv(nn.Module):
    """(Conv → GroupNorm → ReLU) × 2 — 多任务版用 GroupNorm。"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """上采样 + (可选 skip connection) + DoubleConv。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        if x2 is not None:
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]
            x2 = F.pad(
                x2,
                [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
            )
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        return self.conv(x)


class DINOv3_S_UNet_MULTITASK(nn.Module):
    """DINOv3 ViT-S backbone + UNet decoder + 分类头。"""

    def __init__(self, pretrained: bool = True, use_dilation: bool = False):
        super().__init__()
        self.use_dilation = use_dilation
        dino_channels = 384

        # ===== 分割部分 =====
        self.dino = timm.create_model(
            model_name="vit_small_patch16_dinov3.lvd1689m",
            features_only=True,
            pretrained=pretrained,
        )

        self.reduce1 = nn.Conv2d(dino_channels, 128, 1)
        self.reduce2 = nn.Conv2d(dino_channels, 128, 1)
        self.reduce3 = nn.Conv2d(dino_channels, 128, 1)
        self.reduce4 = nn.Conv2d(dino_channels, 128, 1)

        self.up1 = Up(256, 128)
        self.up2 = Up(256, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 128)
        self.head = nn.Conv2d(128, 1, 1)

        if self.use_dilation:
            self.dilate = DilatedConvBlock(128, 128)

        # ===== 分类部分 =====
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.layer_attention = nn.Sequential(
            nn.Linear(dino_channels * 3, 3),
            nn.Softmax(dim=1),
        )

        self.feature_attention = nn.Sequential(
            nn.Linear(dino_channels, dino_channels),
            nn.Sigmoid(),
        )

        classification_feature_dim = dino_channels * 2  # GAP + GMP
        self.classification_backbone = nn.Sequential(
            nn.Linear(classification_feature_dim, 512),
            nn.GroupNorm(8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.benign_malignant_head = nn.Linear(256, 1)
        self.tirads_head = nn.Linear(256, 5)

    def forward(self, x):
        B, C, H, W = x.shape

        # ===== 分割任务 =====
        all_features = self.dino(x)
        features = all_features[-1]

        x1 = F.interpolate(self.reduce1(features), size=(H // 4, W // 4), mode="bilinear")
        x2 = F.interpolate(self.reduce2(features), size=(H // 8, W // 8), mode="bilinear")
        x3 = F.interpolate(self.reduce3(features), size=(H // 16, W // 16), mode="bilinear")
        x4 = F.interpolate(self.reduce4(features), size=(H // 32, W // 32), mode="bilinear")

        if self.use_dilation:
            x4 = self.dilate(x4)

        x = self.up4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        seg_out = F.interpolate(self.head(x), scale_factor=2, mode="bilinear")

        # ===== 分类任务 =====
        all_avg_features = []
        all_max_features = []
        for feat in all_features:
            avg = self.global_avg_pool(feat).view(B, -1)
            max_feat = self.global_max_pool(feat).view(B, -1)
            all_avg_features.append(avg)
            all_max_features.append(max_feat)

        concatenated_avg = torch.cat(all_avg_features, dim=1)
        layer_weights = self.layer_attention(concatenated_avg)

        fused_avg = sum(
            avg * w.unsqueeze(1)
            for avg, w in zip(all_avg_features, layer_weights.transpose(0, 1))
        )
        fused_max = sum(
            mx * w.unsqueeze(1)
            for mx, w in zip(all_max_features, layer_weights.transpose(0, 1))
        )

        attention_weights = self.feature_attention(fused_avg)
        attended_avg = fused_avg * attention_weights
        attended_max = fused_max * attention_weights

        cls_features = torch.cat([attended_avg, attended_max], dim=1)
        cls_features = self.classification_backbone(cls_features)

        benign_malignant = self.benign_malignant_head(cls_features)
        tirads = self.tirads_head(cls_features)

        return seg_out, benign_malignant, tirads
