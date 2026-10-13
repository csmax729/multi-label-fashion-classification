"""Run matched ablations and reserve the test split for the validation winner.

Five configurations use the same prepared split, seed, initial weight source,
and optimization settings. Metrics are measured, never assumed from a resume.
This script uses the current interpreter, so activate the project's environment
or invoke it with ``uv run python scripts/run_experiments.py ...``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Experiment:
    name: str
    loss: str
    augment: bool


EXPERIMENTS = (
    Experiment("baseline_ce", "ce", False),
    Experiment("augmentation_ce", "ce", True),
    Experiment("weighted_ce", "weighted_ce", True),
    Experiment("focal", "focal", True),
    Experiment("label_smoothing", "label_smoothing", True),
)


def _score(report: dict[str, Any], key: str) -> float:
    value = report.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Evaluation report requires a numeric {key}")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be a finite fraction between zero and one")
    return value


def select_validation_winner(reports: dict[str, dict[str, Any]]) -> str:
    """Select by validation macro F1; stable ties follow the supplied run order.

    A test report is rejected even when it happens to have the highest score.
    Accuracy is reported independently and does not determine the winner.
    """
    if not reports:
        raise ValueError("At least one validation report is required")
    for report in reports.values():
        if report.get("split") != "val":
            raise ValueError("Model selection requires only val split reports")
        _score(report, "macro_f1")
        _score(report, "average_task_accuracy")
    return max(reports, key=lambda name: _score(reports[name], "macro_f1"))


def metric_changes(baseline: float, candidate: float) -> dict[str, float | None]:
    """Fractions become percentage points; relative gain uses the baseline.

    For 0.60 -> 0.69 the difference is 9 percentage points and the relative
    gain is 15%. Relative gain is undefined when the baseline is zero.
    """
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (baseline, candidate)):
        raise ValueError("Metrics must be finite fractions between zero and one")
    return {
        "percentage_point_difference": (candidate - baseline) * 100.0,
        "relative_percent_gain": (
            (candidate - baseline) / baseline * 100.0 if baseline > 0.0 else None
        ),
    }


def build_comparison(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Report combined changes and the augmentation-controlled loss effects."""
    select_validation_winner(reports)  # Validate every score and split first.
    baseline = reports["baseline_ce"]
    augmented = reports["augmentation_ce"]
    rows = []
    for experiment in EXPERIMENTS:
        report = reports[experiment.name]
        row: dict[str, Any] = {
            **asdict(experiment),
            "validation_accuracy": _score(report, "average_task_accuracy"),
            "validation_macro_f1": _score(report, "macro_f1"),
            "accuracy_vs_baseline": metric_changes(
                _score(baseline, "average_task_accuracy"),
                _score(report, "average_task_accuracy"),
            ),
            "macro_f1_vs_baseline": metric_changes(
                _score(baseline, "macro_f1"), _score(report, "macro_f1")
            ),
        }
        if experiment.loss != "ce":
            row["accuracy_loss_effect_same_augmentation"] = metric_changes(
                _score(augmented, "average_task_accuracy"),
                _score(report, "average_task_accuracy"),
            )
            row["macro_f1_loss_effect_same_augmentation"] = metric_changes(
                _score(augmented, "macro_f1"), _score(report, "macro_f1")
            )
        rows.append(row)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        result = json.load(file)
    if not isinstance(result, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def preflight(data_dir: Path, *, allow_demo: bool = False) -> dict[str, Any]:
    """Check required inputs without importing training's heavyweight packages."""
    inputs: dict[str, Any] = {}
    for name in ("train.csv", "val.csv", "test.csv", "labels.json"):
        path = data_dir / name
        if not path.is_file():
            raise ValueError(f"Missing prepared dataset file: {path}")
        inputs[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                count = sum(1 for _ in reader)
            if count == 0:
                raise ValueError(f"Dataset split is empty: {path}")
            inputs[name]["rows"] = count
    metadata_path = data_dir / "manifest.json"
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    labels = _read_json(data_dir / "labels.json")
    is_demo = bool(
        metadata.get("dataset_kind") == "demo"
        or metadata.get("is_demo", False)
        or labels.get("is_demo", False)
    )
    if is_demo and not allow_demo:
        raise ValueError(
            "Synthetic demo data requires --allow-demo; it cannot prove fashion accuracy"
        )
    return {
        "data_dir": str(data_dir),
        "inputs": inputs,
        "is_demo": is_demo,
        "dataset_kind": metadata.get("dataset_kind", "unknown"),
        "source": metadata.get("source"),
        "split_seed": metadata.get("seed"),
    }


def train_command(args: argparse.Namespace, experiment: Experiment, destination: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "fashion_classifier.train",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(destination),
        "--loss",
        experiment.loss,
        "--augment" if experiment.augment else "--no-augment",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--image-size",
        str(args.image_size),
        "--seed",
        str(args.seed),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--gamma",
        str(args.gamma),
        "--label-smoothing",
        str(args.label_smoothing),
        "--num-workers",
        str(args.num_workers),
        "--threads",
        str(args.threads),
        "--device",
        args.device,
        "--selection-metric",
        "macro_f1",
        "--pretrained" if args.pretrained else "--no-pretrained",
        "--freeze-backbone" if args.freeze_backbone else "--unfreeze-backbone",
        "--channels-last" if args.channels_last else "--no-channels-last",
    ]
    return command


def evaluate_command(
    args: argparse.Namespace, checkpoint: Path, split: str, output_dir: Path
) -> list[str]:
    if split not in {"val", "test"}:
        raise ValueError("Experiment evaluation is restricted to val or test")
    return [
        sys.executable,
        "-m",
        "fashion_classifier.evaluate",
        "--checkpoint",
        str(checkpoint),
        "--data-dir",
        str(args.data_dir),
        "--split",
        split,
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--threads",
        str(args.threads),
    ]


def _run_command(command: list[str], manifest: dict[str, Any], manifest_path: Path) -> None:
    record: dict[str, Any] = {
        "argv": command,
        "cwd": str(ROOT),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
    }
    manifest["commands"].append(record)
    _write_json(manifest_path, manifest)
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, cwd=ROOT, check=True)
        record["returncode"] = completed.returncode
        record["status"] = "completed"
    except subprocess.CalledProcessError as error:
        record["returncode"] = error.returncode
        record["status"] = "failed"
        raise
    finally:
        record["runtime_seconds"] = time.perf_counter() - started
        _write_json(manifest_path, manifest)


def run_experiments(args: argparse.Namespace) -> Path:
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    data = preflight(args.data_dir, allow_demo=args.allow_demo)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(
            f"Output root contains existing files: {args.output_root}; choose a new --output-root"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "configuration": configuration,
        "dataset": data,
        "experiments": [asdict(experiment) for experiment in EXPERIMENTS],
        "initial_weight_source": (
            "torchvision.ResNet50_Weights.IMAGENET1K_V2"
            if args.pretrained
            else "random_initialization"
        ),
        "selection_metric": "validation macro_f1",
        "commands": [],
        "notes": [
            "All runs use identical seed, prepared splits and shared optimization settings.",
            "Augmentation-only CE versus baseline CE isolates augmentation.",
            "Custom losses versus augmented CE compare losses with augmentation fixed.",
            "Custom losses versus baseline CE compare the combined augmentation/loss changes.",
            "Only the validation winner is evaluated on test; do not tune using test results.",
            "A single fixed-seed comparison does not estimate run-to-run uncertainty.",
            "Demo measurements validate plumbing, not real fashion classification quality.",
        ],
    }
    manifest_path = args.output_root / "run_manifest.json"
    _write_json(manifest_path, manifest)
    started = time.perf_counter()
    validation_reports: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    try:
        for index, experiment in enumerate(EXPERIMENTS, 1):
            print(f"[{index}/{len(EXPERIMENTS)}] Training {experiment.name}", flush=True)
            destination = args.output_root / experiment.name
            _run_command(train_command(args, experiment, destination), manifest, manifest_path)
            summary = _read_json(destination / "summary.json")
            training_config = _read_json(destination / "config.json")
            summary["training_configuration"] = training_config
            summaries[experiment.name] = summary
            manifest.setdefault("resolved_devices", {})[experiment.name] = training_config.get(
                "resolved_device", "unknown"
            )
            checkpoint = destination / "best.pt"
            _run_command(
                evaluate_command(args, checkpoint, "val", destination / "validation"),
                manifest,
                manifest_path,
            )
            validation_reports[experiment.name] = _read_json(
                destination / "validation" / "metrics.json"
            )
        winner = select_validation_winner(validation_reports)
        winner_dir = args.output_root / winner
        comparison: dict[str, Any] = {
            "winner": winner,
            "selection_metric": "validation macro_f1",
            "is_demo": data["is_demo"],
            "rows": build_comparison(validation_reports),
            "training_summaries": summaries,
            "interpretation": manifest["notes"],
        }
        comparison_path = args.output_root / "comparison.json"
        # Persist the validation decision before reading any held-out test metric.
        _write_json(comparison_path, comparison)
        manifest["winner"] = winner
        _write_json(manifest_path, manifest)
        print(f"Validation winner: {winner}; evaluating its held-out test split once", flush=True)
        _run_command(
            evaluate_command(args, winner_dir / "best.pt", "test", winner_dir / "test"),
            manifest,
            manifest_path,
        )
        test_report = _read_json(winner_dir / "test" / "metrics.json")
        if test_report.get("split") != "test":
            raise ValueError("Held-out evaluation report must identify the test split")
        comparison["winner_test_metrics"] = test_report
        _write_json(comparison_path, comparison)
        manifest["status"] = "completed"
        print(f"Comparison saved to {comparison_path}", flush=True)
        return comparison_path
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["runtime_seconds"] = time.perf_counter() - started
        manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(manifest_path, manifest)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    result.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "experiments")
    result.add_argument("--epochs", type=int, default=3)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--image-size", type=int, default=224)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--lr", type=float, default=0.001)
    result.add_argument("--weight-decay", type=float, default=0.0001)
    result.add_argument("--gamma", type=float, default=2.0)
    result.add_argument("--label-smoothing", type=float, default=0.1)
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    result.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use channels-last convolution storage consistently across every training run.",
    )
    result.add_argument("--freeze-backbone", action="store_true", default=True)
    result.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    result.add_argument("--allow-demo", action="store_true")
    return result


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32 or args.threads < 1:
        argument_parser.error("epochs/batch-size/threads must be positive; image-size must be >=32")
    if args.num_workers < 0 or args.lr <= 0 or args.weight_decay < 0 or args.gamma < 0:
        argument_parser.error("Invalid optimizer, loss or worker setting")
    if not 0 <= args.label_smoothing < 1:
        argument_parser.error("label-smoothing must be in [0, 1)")
    try:
        run_experiments(args)
    except (TypeError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        argument_parser.exit(1, f"Experiment run failed: {error}\n")


if __name__ == "__main__":
    main()
