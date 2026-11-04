"""
DINOv3_S_UNet — 纯分割网络。

架构：DINOv3 ViT-S backbone (timm) + UNet decoder。
输入 (B, 3, H, W) → 输出 (B, 1, H, W) pre-sigmoid logits。

注意：DoubleConv 使用 BatchNorm2d（与多任务版的 GroupNorm 不同，
权重不兼容，不能混用）。use_dilation 控制是否在 x4 上加 DilatedConvBlock。
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
    """(Conv → BN → ReLU) × 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
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


class DINOv3_S_UNet(nn.Module):
    """DINOv3 ViT-S backbone + UNet decoder，纯分割。"""

    def __init__(self, pretrained: bool = True, use_dilation: bool = False):
        super().__init__()
        self.use_dilation = use_dilation

        self.dino = timm.create_model(
            model_name="vit_small_patch16_dinov3.lvd1689m",
            features_only=True,
            pretrained=pretrained,
        )

        self.reduce1 = nn.Conv2d(384, 128, 1)
        self.reduce2 = nn.Conv2d(384, 128, 1)
        self.reduce3 = nn.Conv2d(384, 128, 1)
        self.reduce4 = nn.Conv2d(384, 128, 1)

        self.up1 = Up(256, 128)
        self.up2 = Up(256, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 128)
        self.head = nn.Conv2d(128, 1, 1)

        if self.use_dilation:
            self.dilate = DilatedConvBlock(128, 128)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.dino(x)[-1]

        x1 = F.interpolate(self.reduce1(x), size=(H // 4, W // 4), mode="bilinear")
        x2 = F.interpolate(self.reduce2(x), size=(H // 8, W // 8), mode="bilinear")
        x3 = F.interpolate(self.reduce3(x), size=(H // 16, W // 16), mode="bilinear")
        x4 = F.interpolate(self.reduce4(x), size=(H // 32, W // 32), mode="bilinear")

        if self.use_dilation:
            x4 = self.dilate(x4)

        x = self.up4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = F.interpolate(self.head(x), scale_factor=2, mode="bilinear")
        return out
