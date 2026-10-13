"""Evaluate a checkpoint using a held-out split and inspectable per-class metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from fashion_classifier.data import TASKS, FashionDataset, load_labels
from fashion_classifier.losses import IGNORE_INDEX
from fashion_classifier.model import input_memory_format


def _task_metrics(confusion: torch.Tensor, labels: list[str], ignored: int) -> dict:
    true_positives = confusion.diag().double()
    support = confusion.sum(dim=1).double()
    predicted = confusion.sum(dim=0).double()
    denominator = support + predicted
    f1 = torch.where(denominator > 0, 2.0 * true_positives / denominator, 0.0)
    count = int(support.sum())
    return {
        "accuracy": float(true_positives.sum() / count) if count else None,
        # Include every configured class; zero-support classes contribute zero.
        "macro_f1": float(f1.mean()) if count else None,
        "support": count,
        "ignored_targets": ignored,
        "labels": labels,
        "class_support": {label: int(support[i]) for i, label in enumerate(labels)},
        "class_f1": {label: float(f1[i]) for i, label in enumerate(labels)},
        "labels_without_support": [label for i, label in enumerate(labels) if support[i] == 0],
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_axes": {"rows": "true", "columns": "predicted"},
    }


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    labels: dict[str, list[str]],
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict:
    model.eval()
    memory_format = input_memory_format(model)
    confusions = {
        task: torch.zeros((len(labels[task]), len(labels[task])), dtype=torch.int64)
        for task in TASKS
    }
    ignored = dict.fromkeys(TASKS, 0)
    exact_correct = exact_support = images_seen = 0
    accumulated_loss = 0.0
    for images, targets in loader:
        images = images.to(device, memory_format=memory_format)
        targets = {task: values.to(device) for task, values in targets.items()}
        outputs = model(images)
        batch_size = len(images)
        images_seen += batch_size
        if criterion is not None:
            accumulated_loss += float(criterion(outputs, targets)) * batch_size
        all_known = torch.ones(batch_size, dtype=torch.bool, device=device)
        all_correct = torch.ones_like(all_known)
        for task in TASKS:
            predicted = outputs[task].argmax(dim=1)
            truth = targets[task]
            known = truth != IGNORE_INDEX
            ignored[task] += int((~known).sum())
            classes = len(labels[task])
            bins = (truth[known] * classes + predicted[known]).to("cpu")
            confusions[task] += torch.bincount(bins, minlength=classes * classes).reshape(
                classes, classes
            )
            all_known &= known
            all_correct &= predicted == truth
        exact_support += int(all_known.sum())
        exact_correct += int((all_correct & all_known).sum())
    per_task = {
        task: _task_metrics(confusions[task], labels[task], ignored[task]) for task in TASKS
    }
    accuracies = [item["accuracy"] for item in per_task.values() if item["accuracy"] is not None]
    macro_f1 = [item["macro_f1"] for item in per_task.values() if item["macro_f1"] is not None]
    result = {
        "images": images_seen,
        "per_task": per_task,
        "average_task_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
        "macro_f1": sum(macro_f1) / len(macro_f1) if macro_f1 else None,
        "exact_match_accuracy": exact_correct / exact_support if exact_support else None,
        "exact_match_support": exact_support,
        "metric_definitions": {
            "average_task_accuracy": "Unweighted mean of the four task accuracies with known targets.",
            "macro_f1": "Unweighted mean of per-task macro F1 over all training-vocabulary classes; undefined class F1 is zero.",
            "exact_match_accuracy": "Fraction with all four attributes correct, among images whose four targets are known.",
            "unknown_targets": "Labels absent from training vocabulary are ignored per head and reported explicitly.",
        },
    }
    if criterion is not None:
        result["loss"] = accumulated_loss / images_seen if images_seen else None
    return result


def save_evaluation(metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    for task, item in metrics["per_task"].items():
        with (output_dir / f"{task}_confusion.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true / predicted", *item["labels"]])
            for label, row in zip(item["labels"], item["confusion_matrix"]):
                writer.writerow([label, *row])


def main(argv: list[str] | None = None) -> None:
    from fashion_classifier.inference import load_checkpoint, resolve_device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evaluation"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--allow-different-data",
        action="store_true",
        help="Explicitly evaluate another prepared dataset with the same label vocabulary.",
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    labels = checkpoint["labels"]
    if load_labels(args.data_dir / "labels.json") != labels:
        raise ValueError("Dataset label vocabulary differs from the checkpoint")
    expected_hashes = checkpoint["provenance"].get("data_files_sha256", {})
    mismatches = [
        filename
        for filename, expected in expected_hashes.items()
        if not (args.data_dir / filename).is_file()
        or hashlib.sha256((args.data_dir / filename).read_bytes()).hexdigest() != expected
    ]
    if mismatches and not args.allow_different_data:
        raise ValueError(
            f"Prepared data differs from the checkpoint's training provenance: {mismatches}. "
            "Use the original prepared data, or explicitly pass --allow-different-data for an external evaluation."
        )
    dataset = FashionDataset(
        args.data_dir / f"{args.split}.csv",
        labels,
        image_size=checkpoint["config"]["image_size"],
        augment=False,
    )
    if not len(dataset):
        raise ValueError(f"The {args.split} split is empty")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    metrics = evaluate_loader(model, loader, labels, device)
    metrics.update(
        {
            "split": args.split,
            "checkpoint": str(args.checkpoint.resolve()),
            "provenance": checkpoint["provenance"],
            "training_data_matches_checkpoint": not mismatches if expected_hashes else None,
            "training_provenance_file_mismatches": mismatches,
            "evaluated_split_sha256": hashlib.sha256(
                (args.data_dir / f"{args.split}.csv").read_bytes()
            ).hexdigest(),
        }
    )
    save_evaluation(metrics, args.output_dir)
    print(
        json.dumps(
            {
                key: metrics[key]
                for key in [
                    "split",
                    "images",
                    "average_task_accuracy",
                    "macro_f1",
                    "exact_match_accuracy",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
