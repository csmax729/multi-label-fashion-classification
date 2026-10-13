"""Class-balanced categorical losses, with unknown targets safely ignored."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from fashion_classifier.data import TASKS

IGNORE_INDEX = -100


def balanced_class_weights(counts: Sequence[int]) -> torch.Tensor:
    """Return N/(K*n_c), calculated from training labels alone.

    Missing training classes receive zero weight; an all-empty head is invalid.
    Data preparation normally ensures every configured class occurs in training.
    """
    values = torch.as_tensor(counts, dtype=torch.float32)
    if values.ndim != 1 or not len(values) or (values < 0).any() or values.sum() == 0:
        raise ValueError("Class counts must be a nonempty vector with a positive total")
    weights = torch.zeros_like(values)
    present = values > 0
    weights[present] = values.sum() / (present.sum() * values[present])
    return weights


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        known = targets != IGNORE_INDEX
        if not known.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[known], targets[known], weight=self.weight)


class FocalLoss(nn.Module):
    """Multiclass focal loss: -w_y (1-p_y)^gamma log(p_y)."""

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        known = targets != IGNORE_INDEX
        if not known.any():
            return logits.sum() * 0.0
        targets = targets[known]
        log_probability = (
            F.log_softmax(logits[known], dim=-1).gather(1, targets[:, None]).squeeze(1)
        )
        losses = -(1.0 - log_probability.exp()).pow(self.gamma) * log_probability
        if self.weight is None:
            return losses.mean()
        weights = self.weight[targets]
        return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


class LabelSmoothedCrossEntropyLoss(nn.Module):
    """Target-class balanced cross entropy against a smoothed distribution.

    Smoothing distributes epsilon uniformly across every class, including the
    target. The target class's training weight scales the complete example.
    """

    def __init__(self, weight: torch.Tensor | None = None, smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.register_buffer("weight", weight)
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        known = targets != IGNORE_INDEX
        if not known.any():
            return logits.sum() * 0.0
        targets = targets[known]
        log_probabilities = F.log_softmax(logits[known], dim=-1)
        negative_log_likelihood = -log_probabilities.gather(1, targets[:, None]).squeeze(1)
        uniform_loss = -log_probabilities.mean(dim=-1)
        losses = (1.0 - self.smoothing) * negative_log_likelihood + self.smoothing * uniform_loss
        if self.weight is None:
            return losses.mean()
        weights = self.weight[targets]
        return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


class MultiTaskLoss(nn.Module):
    """Give each head equal influence, independently of its class cardinality."""

    def __init__(
        self,
        name: str,
        counts: Mapping[str, Sequence[int]],
        *,
        gamma: float = 2.0,
        smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        if name not in {"ce", "weighted_ce", "focal", "label_smoothing"}:
            raise ValueError(f"Unknown loss: {name}")
        if set(counts) != set(TASKS):
            raise ValueError(f"Class counts must contain exactly {TASKS}")
        self.criteria = nn.ModuleDict()
        for task in TASKS:
            weights = balanced_class_weights(counts[task])
            if name == "ce":
                criterion = WeightedCrossEntropyLoss()
            elif name == "weighted_ce":
                criterion = WeightedCrossEntropyLoss(weights)
            elif name == "focal":
                criterion = FocalLoss(weights, gamma=gamma)
            else:
                criterion = LabelSmoothedCrossEntropyLoss(weights, smoothing=smoothing)
            self.criteria[task] = criterion

    def forward(
        self, logits: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        losses = [
            self.criteria[task](logits[task], targets[task])
            for task in TASKS
            if (targets[task] != IGNORE_INDEX).any()
        ]
        if not losses:
            return sum(tensor.sum() * 0.0 for tensor in logits.values())
        return torch.stack(losses).mean()
