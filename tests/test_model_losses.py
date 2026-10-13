"""Numerical behavior, ignored targets, and the four-head architecture contract."""

import pytest
import torch
from torch.nn import functional as F

from fashion_classifier.data import TASKS
from fashion_classifier.evaluate import _task_metrics, evaluate_loader
from fashion_classifier.losses import (
    FocalLoss,
    LabelSmoothedCrossEntropyLoss,
    MultiTaskLoss,
    WeightedCrossEntropyLoss,
    balanced_class_weights,
)
from fashion_classifier.model import FashionMultiTaskModel, input_memory_format


@pytest.mark.parametrize(
    "loss_type", [WeightedCrossEntropyLoss, FocalLoss, LabelSmoothedCrossEntropyLoss]
)
def test_losses_propagate_finite_gradients_and_ignore_unknown_targets(loss_type):
    logits = torch.tensor(
        [[2.0, -1.0, 0.5], [0.2, 0.4, -0.5], [-1.0, 0.2, 2.0]], requires_grad=True
    )
    loss = loss_type(balanced_class_weights([6, 3, 2]))(logits, torch.tensor([0, -100, 2]))
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[1], torch.zeros(3))
    assert logits.grad[0].abs().sum() > 0
    assert logits.grad[2].abs().sum() > 0


@pytest.mark.parametrize(
    "loss_type", [WeightedCrossEntropyLoss, FocalLoss, LabelSmoothedCrossEntropyLoss]
)
def test_all_unknown_targets_have_zero_loss_and_gradient(loss_type):
    logits = torch.randn(2, 3, requires_grad=True)
    loss = loss_type()(logits, torch.full((2,), -100))
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_custom_losses_match_cross_entropy_at_neutral_settings():
    logits = torch.tensor([[1.0, -1.0], [-0.5, 0.5]])
    targets = torch.tensor([0, 1])
    expected = F.cross_entropy(logits, targets)
    assert torch.allclose(FocalLoss(gamma=0)(logits, targets), expected)
    assert torch.allclose(LabelSmoothedCrossEntropyLoss(smoothing=0)(logits, targets), expected)
    assert FocalLoss(gamma=2)(logits, targets) < expected
    weights = torch.tensor([1.0, 3.0])
    weighted_expected = F.cross_entropy(logits, targets, weight=weights)
    assert torch.allclose(FocalLoss(weight=weights, gamma=0)(logits, targets), weighted_expected)
    assert torch.allclose(
        LabelSmoothedCrossEntropyLoss(weight=weights, smoothing=0)(logits, targets),
        weighted_expected,
    )


def test_class_weights_use_inverse_training_frequency():
    weights = balanced_class_weights([30, 10, 0])
    assert weights[1] / weights[0] == pytest.approx(3)
    assert weights[2] == 0
    with pytest.raises(ValueError):
        balanced_class_weights([0, 0])


@pytest.mark.parametrize("name", ["ce", "weighted_ce", "focal", "label_smoothing"])
def test_multitask_losses_update_every_head(name):
    outputs = {task: torch.randn(3, 2, requires_grad=True) for task in TASKS}
    targets = {task: torch.tensor([0, 1, 0]) for task in TASKS}
    loss = MultiTaskLoss(name, {task: [2, 1] for task in TASKS})(outputs, targets)
    loss.backward()
    assert all(
        torch.isfinite(output.grad).all() and output.grad.abs().sum() > 0
        for output in outputs.values()
    )


def test_resnet_has_four_categorical_heads_and_frozen_batchnorm():
    torch.set_num_threads(2)
    sizes = {task: index + 2 for index, task in enumerate(TASKS)}
    model = FashionMultiTaskModel(sizes, pretrained=False, freeze_backbone=True)
    model.train()
    assert model.training and not model.backbone.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    outputs = model(torch.randn(2, 3, 32, 32))
    assert {task: list(value.shape) for task, value in outputs.items()} == {
        task: [2, count] for task, count in sizes.items()
    }
    sum(value.sum() for value in outputs.values()).backward()
    assert all(head.weight.grad is not None for head in model.heads.values())
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_metric_definition_includes_unsupported_classes_explicitly():
    metrics = _task_metrics(torch.tensor([[2, 0], [0, 0]]), ["common", "absent"], ignored=1)
    assert metrics["accuracy"] == 1
    assert metrics["macro_f1"] == 0.5
    assert metrics["labels_without_support"] == ["absent"]
    assert metrics["ignored_targets"] == 1


def test_channels_last_preserves_categorical_outputs():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = FashionMultiTaskModel({task: 2 for task in TASKS}, pretrained=False).eval()
    images = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        contiguous_outputs = model(images)
        model.to(memory_format=torch.channels_last)
        assert input_memory_format(model) == torch.channels_last
        channel_outputs = model(images.to(memory_format=input_memory_format(model)))
    for task in TASKS:
        assert torch.allclose(contiguous_outputs[task], channel_outputs[task], rtol=1e-4, atol=1e-5)


def test_evaluation_distinguishes_task_accuracy_from_exact_match():
    class FixedPredictions(torch.nn.Module):
        def forward(self, images):
            indices = images[:, 0].long()
            first = torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])[indices]
            other = torch.tensor([[3.0, 0.0]]).repeat(len(indices), 1)
            return {task: first if task == TASKS[0] else other for task in TASKS}

    examples = []
    for index in range(3):
        targets = {task: torch.tensor(0) for task in TASKS}
        if index == 2:
            targets[TASKS[-1]] = torch.tensor(-100)
        examples.append((torch.tensor([float(index)]), targets))
    loader = torch.utils.data.DataLoader(examples, batch_size=2)
    metrics = evaluate_loader(
        FixedPredictions(), loader, {task: ["a", "b"] for task in TASKS}, torch.device("cpu")
    )
    assert metrics["average_task_accuracy"] == pytest.approx(11 / 12)
    assert metrics["exact_match_accuracy"] == 0.5
    assert metrics["exact_match_support"] == 2
    assert metrics["per_task"][TASKS[-1]]["ignored_targets"] == 1
