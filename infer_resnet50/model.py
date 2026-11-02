"""ResNet-50 binary classifier (inference / training shared)."""

import torch.nn as nn
from torchvision import models


def build_model(device, pretrained: bool = False) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    model.to(device)
    return model
