"""Fit four categorical heads using training data and validation-only selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from fashion_classifier.data import TASKS, FashionDataset, load_labels
from fashion_classifier.evaluate import evaluate_loader
from fashion_classifier.inference import load_checkpoint, resolve_device
from fashion_classifier.losses import MultiTaskLoss
from fashion_classifier.model import ARCHITECTURE, FashionMultiTaskModel


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def training_class_counts(path: Path, labels: dict[str, list[str]]) -> dict[str, list[int]]:
    """Read label frequencies without decoding images or looking at val/test."""
    indices = {task: {label: i for i, label in enumerate(labels[task])} for task in TASKS}
    counts = {task: [0] * len(labels[task]) for task in TASKS}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for task in TASKS:
                if row[task] not in indices[task]:
                    raise ValueError(f"Training label {row[task]!r} absent from {task} vocabulary")
                counts[task][indices[task][row[task]]] += 1
    if any(sum(values) == 0 for values in counts.values()):
        raise ValueError("Training split is empty")
    return counts


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _provenance(data_dir: Path, model: FashionMultiTaskModel, config: dict) -> dict:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    kind = manifest.get("dataset_kind", "unverified")
    return {
        "demo": kind == "demo",
        "dataset_kind": kind,
        "dataset_source": manifest.get("source", "unverified local data"),
        "backbone_weight_source": model.backbone_weight_source,
        "channels_last": config["channels_last"],
        "seed": config["seed"],
        "training_class_counts_source": "train.csv only",
        "checkpoint_selection_split": "val",
        "test_used_during_training": False,
        "torch_version": str(torch.__version__),
        "data_files_sha256": {
            filename: hashlib.sha256((data_dir / filename).read_bytes()).hexdigest()
            for filename in ("train.csv", "val.csv", "labels.json")
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--loss", choices=["ce", "weighted_ce", "focal", "label_smoothing"], default="weighted_ce"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print flushed JSON training progress every N batches and at the final batch; 0 disables it.",
    )
    parser.add_argument(
        "--selection-metric", choices=["macro_f1", "average_task_accuracy"], default="macro_f1"
    )
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use channels-last convolution storage; can improve CPU throughput.",
    )
    backbone = parser.add_mutually_exclusive_group()
    backbone.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true")
    backbone.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=True)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Initialize model weights, validate vocabulary, and start a fresh optimizer.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or args.num_workers < 0:
        parser.error(
            "epochs, batch-size, threads must be positive; num-workers must be nonnegative"
        )
    if not 32 <= args.image_size <= 2048:
        parser.error("image-size must be between 32 and 2048")
    if args.log_every < 0:
        parser.error("log-every must be nonnegative")
    if args.lr <= 0 or args.weight_decay < 0 or args.gamma < 0 or not 0 <= args.label_smoothing < 1:
        parser.error("Require lr > 0, weight-decay/gamma >= 0, and 0 <= label-smoothing < 1")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    labels = load_labels(args.data_dir / "labels.json")
    counts = training_class_counts(args.data_dir / "train.csv", labels)
    # Initialize before data augmentation consumes RNG: ablations with the same
    # seed start from identical backbone and head weights.
    if args.init_checkpoint:
        model, initial = load_checkpoint(args.init_checkpoint, device)
        if initial["labels"] != labels:
            raise ValueError(
                "Initialization checkpoint label vocabulary differs from training data"
            )
        model.backbone_weight_source = initial["provenance"].get(
            "backbone_weight_source", "checkpoint"
        )
        model.set_backbone_trainable(not args.freeze_backbone)
    else:
        model = FashionMultiTaskModel(
            {task: len(labels[task]) for task in TASKS},
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
        ).to(device)
    memory_format = torch.channels_last if args.channels_last else torch.contiguous_format
    model.to(memory_format=memory_format)
    train_data = FashionDataset(
        args.data_dir / "train.csv", labels, image_size=args.image_size, augment=args.augment
    )
    val_data = FashionDataset(
        args.data_dir / "val.csv", labels, image_size=args.image_size, augment=False
    )
    if not len(train_data) or not len(val_data):
        raise ValueError("Training and validation splits must both contain images")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
        worker_init_fn=_seed_worker,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    criterion = MultiTaskLoss(
        args.loss, counts, gamma=args.gamma, smoothing=args.label_smoothing
    ).to(device)
    optimizer = Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["resolved_device"] = str(device)
    config["optimizer"] = "Adam"
    config["optimizer_state_resumed"] = False
    provenance = _provenance(args.data_dir, model, config)
    if args.init_checkpoint:
        provenance["initialized_from_checkpoint"] = str(args.init_checkpoint.resolve())
    _write_json(args.output_dir / "config.json", config)
    _write_json(args.output_dir / "training_class_counts.json", counts)
    history: list[dict] = []
    best_score = float("-inf")
    summary = {}
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        accumulated_loss = 0.0
        examples = 0
        total_batches = len(train_loader)
        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, memory_format=memory_format)
            targets = {task: values.to(device) for task, values in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; inspect data and learning rate")
            loss.backward()
            optimizer.step()
            accumulated_loss += float(loss.detach()) * len(images)
            examples += len(images)
            if args.log_every and (
                batch_index % args.log_every == 0 or batch_index == total_batches
            ):
                progress = {
                    "event": "training_progress",
                    "epoch": epoch,
                    "batch": batch_index,
                    "total_batches": total_batches,
                    "images_seen": examples,
                    "train_loss_so_far": accumulated_loss / examples,
                    "elapsed_seconds": time.perf_counter() - start,
                }
                print(json.dumps(progress, allow_nan=False), flush=True)
        validation = evaluate_loader(model, val_loader, labels, device, criterion)
        score = validation[args.selection_metric]
        if score is None:
            raise ValueError("No known validation labels; checkpoint selection is impossible")
        record = {
            "epoch": epoch,
            "train_loss": accumulated_loss / examples,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["average_task_accuracy"],
            "macro_f1": validation["macro_f1"],
            "exact_match_accuracy": validation["exact_match_accuracy"],
            "seconds": time.perf_counter() - start,
        }
        history.append(record)
        if score > best_score:
            best_score = score
            checkpoint_path = args.output_dir / "best.pt"
            checkpoint = {
                "schema_version": 1,
                "architecture": ARCHITECTURE,
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "labels": labels,
                "config": config,
                "provenance": provenance,
                "epoch": epoch,
                "validation_metrics": validation,
            }
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_path)
            summary = {
                "best_epoch": epoch,
                "best_validation_accuracy": validation["average_task_accuracy"],
                "best_macro_f1": validation["macro_f1"],
                "selection_metric": args.selection_metric,
                "best_selection_score": score,
                "checkpoint": str(checkpoint_path.resolve()),
                "provenance": provenance,
            }
            _write_json(args.output_dir / "summary.json", summary)
        _write_json(args.output_dir / "history.json", history)
        print(json.dumps(record, allow_nan=False), flush=True)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
