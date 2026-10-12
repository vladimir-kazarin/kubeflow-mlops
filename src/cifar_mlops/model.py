"""ResNet-18 classifier that takes raw uint8 images, so training and serving share one preprocessing path."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from cifar_mlops.data import LABELS_FILE

IMAGE_SIZE = 128
NUM_CLASSES = 10
MODEL_FILE = "model.pt"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ServingModel(nn.Module):
    """uint8 images (N, 3, H, W) -> logits (N, NUM_CLASSES); resizing and normalization happen inside."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.image_size = IMAGE_SIZE  # attribute, not global: TorchScript cannot read module constants
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.float() / 255.0
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.backbone(x)


def build_model(pretrained: bool) -> ServingModel:
    """ResNet-18 with a 10-class head; pretrained=True starts from ImageNet weights."""
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    backbone.fc = nn.Linear(backbone.fc.in_features, NUM_CLASSES)
    return ServingModel(backbone)


def export_model(model: nn.Module, labels: Sequence[str], output_dir: Path) -> None:
    """Write a CPU TorchScript model and its class labels; this directory is what KServe serves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model.to("cpu").eval())
    scripted.save(str(output_dir / MODEL_FILE))
    (output_dir / LABELS_FILE).write_text(json.dumps(list(labels)))


def load_exported(model_dir: Path) -> tuple[torch.jit.ScriptModule, list[str]]:
    model_path, labels_path = model_dir / MODEL_FILE, model_dir / LABELS_FILE
    for path in (model_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(f"Exported model is missing {path}")
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    return model, json.loads(labels_path.read_text())
