"""Audited preparation and PyTorch loading for the four fashion attributes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

TASKS = ("gender", "articleType", "baseColour", "season")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_labels(path: str | Path) -> dict[str, list[str]]:
    labels = json.loads(Path(path).read_text())
    if set(labels) != set(TASKS):
        raise ValueError(f"labels.json must contain exactly {TASKS}")
    for task in TASKS:
        classes = labels[task]
        if not isinstance(classes, list) or not classes or len(classes) != len(set(classes)):
            raise ValueError(f"Invalid or empty class vocabulary for {task}")
        if any(not isinstance(value, str) or not value for value in classes):
            raise ValueError(f"Class names for {task} must be nonempty strings")
    return labels


def get_transforms(image_size: int = 224, augment: bool = False):
    """ImageNet preprocessing shared by training and inference; six train augmentations."""
    from torchvision import transforms

    if image_size < 16:
        raise ValueError("image_size must be at least 16")
    if augment:
        operations = [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]
    else:
        operations = [transforms.Resize((image_size, image_size))]
    operations += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if augment:
        operations.append(transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)))
    return transforms.Compose(operations)


make_transforms = get_transforms


class FashionDataset:
    """Returns (RGB image tensor, task -> class index); unknown labels use -100."""

    def __init__(self, csv_path, labels, image_size=224, augment=False):
        self.labels = load_labels(labels) if isinstance(labels, (str, Path)) else labels
        self.encoders = {
            task: {name: i for i, name in enumerate(self.labels[task])} for task in TASKS
        }
        with Path(csv_path).open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        self.transforms = get_transforms(image_size, augment)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch

        row = self.rows[index]
        with Image.open(row["image_path"]) as source:
            tensor = self.transforms(source.convert("RGB"))
        targets = {
            task: torch.tensor(self.encoders[task].get(row[task], -100), dtype=torch.long)
            for task in TASKS
        }
        return tensor, targets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_styles(path: Path, audit: dict) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or len(set(header)) != len(header) or not {"id", *TASKS}.issubset(header):
            raise ValueError(f"styles.csv requires unique columns id and {TASKS}")
        for fields in reader:
            audit["source_rows"] += 1
            line = reader.line_num
            # The original Kaggle CSV has unquoted commas in its final display-name field.
            if len(fields) > len(header) and header[-1] == "productDisplayName":
                fields = fields[: len(header) - 1] + [",".join(fields[len(header) - 1 :])]
                audit["repaired_display_name_rows"] += 1
                audit["repaired_line_examples"] = (audit["repaired_line_examples"] + [line])[:20]
            if len(fields) != len(header):
                audit["excluded"]["malformed_csv"] += 1
                audit["malformed_examples"] = (
                    audit["malformed_examples"] + [{"line": line, "fields": len(fields)}]
                )[:20]
                continue
            row = {key: value.strip() for key, value in zip(header, fields)}
            if not row["id"] or Path(row["id"]).name != row["id"] or row["id"] in {".", ".."}:
                audit["excluded"]["invalid_id"] += 1
                continue
            if any(not row[task] or row[task].lower() in {"nan", "null", "none"} for task in TASKS):
                audit["excluded"]["missing_task_label"] += 1
                continue
            row["_line"] = line
            rows.append(row)
    return rows


def _split_groups(rows: list[dict], seed: int, val_size: float, test_size: float):
    """Greedy group-stratification with train coverage for every attribute label.

    Exact decoded RGB image duplicates share a group. A deterministic set cover first
    reserves training groups so every label is trainable; remaining groups balance
    the four marginal class distributions and split sizes. No evaluation row is
    dropped merely for being rare. Rare labels can be absent from evaluation.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["_hash"]].append(row)
    if len(grouped) < 3:
        raise ValueError(
            "At least three distinct image-content groups are required for train/val/test"
        )
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    rank = {key: i for i, key in enumerate(keys)}
    label_sets = {
        key: {(task, row[task]) for row in group for task in TASKS}
        for key, group in grouped.items()
    }
    all_labels = set().union(*label_sets.values())
    support = Counter(label for labels in label_sets.values() for label in labels)
    # Single-group labels force their whole group into training. Reserve these
    # together to avoid a quadratic set-cover loop on many singleton classes.
    assignments = {
        key: "train" for key in keys if any(support[label] == 1 for label in label_sets[key])
    }
    covered = set().union(*(label_sets[key] for key in assignments)) if assignments else set()
    uncovered = set(all_labels) - covered
    while uncovered:
        available = [key for key in keys if key not in assignments]
        best = max(
            available,
            key=lambda key: (
                math.fsum(1 / support[label] for label in sorted(label_sets[key] & uncovered)),
                -rank[key],
            ),
        )
        assignments[best] = "train"
        uncovered -= label_sets[best]

    fractions = {"train": 1 - val_size - test_size, "val": val_size, "test": test_size}
    totals = Counter((task, row[task]) for row in rows for task in TASKS)
    counts = {split: Counter() for split in fractions}
    sizes = Counter()
    for key, split in assignments.items():
        counts[split].update((task, row[task]) for row in grouped[key] for task in TASKS)
        sizes[split] += len(grouped[key])
    remaining = sorted(
        (key for key in keys if key not in assignments),
        key=lambda key: (
            -math.fsum(1 / support[label] for label in sorted(label_sets[key])),
            rank[key],
        ),
    )
    for key in remaining:
        additions = Counter((task, row[task]) for row in grouped[key] for task in TASKS)
        group_size = len(grouped[key])

        def change_in_cost(split, group_size=group_size, additions=additions):
            desired_size = len(rows) * fractions[split]
            before_size = sizes[split] - desired_size
            score = 4 * ((before_size + group_size) ** 2 - before_size**2) / (desired_size + 1)
            for label, amount in additions.items():
                desired = totals[label] * fractions[split]
                before = counts[split][label] - desired
                score += ((before + amount) ** 2 - before**2) / (desired + 1)
            return score

        split = min(fractions, key=change_in_cost)
        assignments[key] = split
        counts[split].update(additions)
        sizes[split] += group_size

    # Repair empty evaluation splits only when train label coverage remains valid.
    for split in ("val", "test"):
        if sizes[split]:
            continue
        candidates = []
        for key in keys:
            origin = assignments[key]
            additions = Counter((task, row[task]) for row in grouped[key] for task in TASKS)
            if sizes[origin] <= len(grouped[key]):
                continue
            if origin == "train" and any(
                counts["train"][label] <= amount for label, amount in additions.items()
            ):
                continue
            candidates.append(key)
        if not candidates:
            raise ValueError(
                "Cannot create three nonempty leakage-safe splits while keeping every label in training; provide more samples"
            )
        key = min(candidates, key=lambda key: (len(grouped[key]), rank[key]))
        origin = assignments[key]
        additions = Counter((task, row[task]) for row in grouped[key] for task in TASKS)
        counts[origin].subtract(additions)
        counts[split].update(additions)
        sizes[origin] -= len(grouped[key])
        sizes[split] += len(grouped[key])
        assignments[key] = split
    splits = {split: [] for split in fractions}
    for key in keys:
        splits[assignments[key]].extend(grouped[key])
    return splits, support


def prepare_data(styles, images_dir, output_dir, seed=42, val_size=0.15, test_size=0.15):
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than 1")
    styles, images_dir, output_dir = (
        Path(styles).resolve(),
        Path(images_dir).resolve(),
        Path(output_dir),
    )
    audit = {
        "source_rows": 0,
        "repaired_display_name_rows": 0,
        "repaired_line_examples": [],
        "malformed_examples": [],
        "excluded": Counter(),
        "invalid_image_examples": [],
    }
    source_rows = _read_styles(styles, audit)
    id_rows = defaultdict(list)
    for row in source_rows:
        id_rows[row["id"]].append(row)
    unique_rows = []
    for duplicates in id_rows.values():
        if len({tuple(row[task] for task in TASKS) for row in duplicates}) > 1:
            audit["excluded"]["conflicting_duplicate_id"] += len(duplicates)
        else:
            unique_rows.append(duplicates[0])
            audit["excluded"]["identical_duplicate_id"] += len(duplicates) - 1
    image_index = defaultdict(list)
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_index[path.stem].append(path)
    usable_rows = []
    for row in unique_rows:
        paths = image_index[row["id"]]
        if not paths:
            audit["excluded"]["missing_image"] += 1
            continue
        if len(paths) != 1:
            audit["excluded"]["ambiguous_image_filename"] += 1
            continue
        path = paths[0]
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                rgb_image = image.convert("RGB")
                rgb_image.load()
                pixel_hash = hashlib.sha256(f"RGB:{rgb_image.width}x{rgb_image.height}:".encode())
                pixel_hash.update(rgb_image.tobytes())
        except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            audit["excluded"]["invalid_image"] += 1
            audit["invalid_image_examples"] = (
                audit["invalid_image_examples"] + [{"id": row["id"], "reason": str(error)}]
            )[:20]
            continue
        row["image_path"] = str(path.resolve())
        row["_hash"] = pixel_hash.hexdigest()
        usable_rows.append(row)
    splits, support = _split_groups(usable_rows, seed, val_size, test_size)
    labels = {task: sorted({row[task] for row in splits["train"]}) for task in TASKS}
    digest_groups = defaultdict(list)
    for row in usable_rows:
        digest_groups[row["_hash"]].append(row)
    audit.update(
        {
            "usable_rows": len(usable_rows),
            "distinct_image_groups": len(digest_groups),
            "duplicate_content_groups": sum(len(group) > 1 for group in digest_groups.values()),
            "duplicate_content_rows": sum(len(group) - 1 for group in digest_groups.values()),
            "conflicting_content_label_groups": sum(
                len({tuple(row[task] for task in TASKS) for row in group}) > 1
                for group in digest_groups.values()
            ),
            "split_counts": {split: len(rows) for split, rows in splits.items()},
            "unknown_labels": {
                split: {
                    task: dict(Counter(row[task] for row in rows if row[task] not in labels[task]))
                    for task in TASKS
                }
                for split, rows in splits.items()
            },
            "singleton_group_labels": {
                task: [
                    label
                    for (label_task, label), count in sorted(support.items())
                    if label_task == task and count == 1
                ]
                for task in TASKS
            },
            "evaluation_absent_labels": {
                split: {
                    task: sorted(set(labels[task]) - {row[task] for row in rows}) for task in TASKS
                }
                for split, rows in splits.items()
                if split != "train"
            },
            "split_strategy": "Deterministic decoded-RGB-SHA256 group split; reserve train label coverage, then greedily balance four marginal class distributions and sizes. Fractions are approximate. Singleton group labels are training-only; evaluation has less class coverage, reported explicitly.",
            "leakage_checks": {
                "id_overlap": False,
                "exact_image_hash_overlap": False,
                "hash_definition": "SHA256 of RGB dimensions and decoded pixel bytes",
                "near_duplicate_detection": "not performed; perceptual duplicates require a separate audit",
            },
        }
    )
    ids = [{row["id"] for row in rows} for rows in splits.values()]
    hashes = [{row["_hash"] for row in rows} for rows in splits.values()]
    if any(ids[i] & ids[j] or hashes[i] & hashes[j] for i in range(3) for j in range(i)):
        raise AssertionError("Split leakage detected")
    provenance = {"dataset_kind": "user", "source": str(styles)}
    for marker_name in ("demo_metadata.json", "download_metadata.json"):
        marker = styles.parent / marker_name
        if marker.is_file():
            provenance = json.loads(marker.read_text())
            break
    audit["provenance"] = provenance
    audit["source_csv_sha256"] = _sha256(styles)
    audit["seed"] = seed
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with (output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "image_path", *TASKS])
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    (output_dir / "quality_report.json").write_text(json.dumps(audit, indent=2) + "\n")
    manifest = {
        **provenance,
        "split_counts": audit["split_counts"],
        "seed": seed,
        "quality_report": "quality_report.json",
        "tasks": list(TASKS),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--styles", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default="data/processed", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--val-size", default=0.15, type=float)
    parser.add_argument("--test-size", default=0.15, type=float)
    args = parser.parse_args()
    report = prepare_data(
        args.styles, args.images_dir, args.output_dir, args.seed, args.val_size, args.test_size
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("source_rows", "usable_rows", "excluded", "split_counts", "provenance")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
