"""
Three architectures for lung CT 3-class classification (Normal / Benign / Malignant):

1. CNNBaseline      — trained from scratch, gives you an honest "no pretraining"
                       reference point to show transfer learning actually helps.
2. build_resnet50   — ImageNet-pretrained ResNet50, fine-tuned.
3. build_efficientnet_b0 — ImageNet-pretrained EfficientNet-B0, fine-tuned.

All three expose the same interface: input (B, 3, 224, 224) -> logits (B, NUM_CLASSES).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import NUM_CLASSES


class CNNBaseline(nn.Module):
    """
    A from-scratch CNN — 4 conv blocks + global average pool + classifier head.
    Deliberately simple: this is the baseline everything else has to beat.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),      # 224 -> 112
            conv_block(32, 64),     # 112 -> 56
            conv_block(64, 128),    # 56 -> 28
            conv_block(128, 256),   # 28 -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def get_target_layer(self):
        """Last conv block — used as the Grad-CAM target layer (Module 3)."""
        return self.features[-1][3]  # last Conv2d in the final conv_block


def build_resnet50(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_efficientnet_b0(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_target_layer(model: nn.Module, model_name: str):
    """
    Returns the conv layer Grad-CAM (Module 3) should hook into, per architecture.
    Centralized here so the explainability module doesn't need to know internals
    of each backbone.
    """
    if model_name == "cnn_baseline":
        return model.get_target_layer()
    elif model_name == "resnet50":
        return model.layer4[-1]
    elif model_name == "efficientnet_b0":
        return model.features[-1]
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def build_model(model_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    if model_name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes)
    elif model_name == "resnet50":
        return build_resnet50(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "efficientnet_b0":
        return build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Expected one of: "
            "cnn_baseline, resnet50, efficientnet_b0"
        )


if __name__ == "__main__":
    # Quick shape sanity check for all three architectures
    dummy = torch.randn(2, 3, 224, 224)
    for name in ["cnn_baseline", "resnet50", "efficientnet_b0"]:
        m = build_model(name)
        out = m(dummy)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"{name:<18} output={tuple(out.shape)}  params={n_params:,}")
