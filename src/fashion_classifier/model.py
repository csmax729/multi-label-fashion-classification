"""A shared ResNet50 with one categorical head for each fashion attribute.

Although a product has several labels, each attribute is mutually exclusive.
The heads therefore output logits for softmax/cross entropy, not sigmoid/BCE.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from fashion_classifier.data import TASKS

ARCHITECTURE = "resnet50_shared_backbone_four_heads"


def input_memory_format(model: nn.Module) -> torch.memory_format:
    """Match convolution input layout to the model's stored weight layout."""
    for parameter in model.parameters():
        if parameter.ndim == 4:
            return (
                torch.channels_last
                if parameter.is_contiguous(memory_format=torch.channels_last)
                else torch.contiguous_format
            )
    return torch.contiguous_format


class FashionMultiTaskModel(nn.Module):
    """ResNet50's 2,048 dimensional feature vector feeds four linear heads."""

    def __init__(
        self,
        class_counts: Mapping[str, int],
        *,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if set(class_counts) != set(TASKS):
            raise ValueError(f"class_counts must contain exactly {TASKS}")
        if any(count < 1 for count in class_counts.values()):
            raise ValueError("Every classification head requires at least one class")
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        feature_count = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleDict(
            {task: nn.Linear(feature_count, class_counts[task]) for task in TASKS}
        )
        self.backbone_weight_source = (
            "torchvision.ResNet50_Weights.IMAGENET1K_V2" if pretrained else "random_initialization"
        )
        self.freeze_backbone = False
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.freeze_backbone = not trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
        self.backbone.train(self.training and trainable)

    def train(self, mode: bool = True) -> FashionMultiTaskModel:
        super().train(mode)
        # Frozen batch-normalization statistics must stay fixed during head warmup.
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        return {task: self.heads[task](features) for task in TASKS}
