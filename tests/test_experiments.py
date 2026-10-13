"""Protect experiment selection, gain arithmetic, and held-out data boundaries."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_experiments.py"
SPEC = importlib.util.spec_from_file_location("fashion_experiment_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def report(accuracy: float, macro_f1: float, *, split: str = "val") -> dict:
    return {"split": split, "average_task_accuracy": accuracy, "macro_f1": macro_f1}


def prepared_data(tmp_path: Path, *, is_demo: bool = False) -> Path:
    destination = tmp_path / "prepared"
    destination.mkdir()
    for split, identifier in (("train", "1"), ("val", "2"), ("test", "3")):
        (destination / f"{split}.csv").write_text(f"id,image_path\n{identifier},image.jpg\n")
    (destination / "labels.json").write_text(json.dumps({"tasks": {}}))
    (destination / "manifest.json").write_text(
        json.dumps({"dataset_kind": "demo" if is_demo else "public", "seed": 42})
    )
    return destination


def test_percent_gain_and_percentage_points_are_distinct():
    changes = runner.metric_changes(0.60, 0.69)
    assert changes["relative_percent_gain"] == pytest.approx(15)
    assert changes["percentage_point_difference"] == pytest.approx(9)


def test_zero_baseline_has_no_defined_relative_gain():
    assert runner.metric_changes(0, 0.25) == {
        "percentage_point_difference": 25,
        "relative_percent_gain": None,
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_metric_is_rejected(value):
    with pytest.raises(ValueError, match="finite fractions"):
        runner.metric_changes(0.5, value)


def test_winner_is_selected_by_validation_macro_f1_not_accuracy():
    scores = {"high_accuracy": report(0.91, 0.61), "high_f1": report(0.82, 0.79)}
    assert runner.select_validation_winner(scores) == "high_f1"


def test_test_score_cannot_participate_in_selection():
    scores = {"validation": report(0.82, 0.79), "test": report(0.99, 0.99, split="test")}
    with pytest.raises(ValueError, match="only val"):
        runner.select_validation_winner(scores)


def test_tied_validation_scores_use_stable_run_order():
    assert (
        runner.select_validation_winner({"first": report(0.7, 0.8), "second": report(0.9, 0.8)})
        == "first"
    )


def test_comparison_distinguishes_combined_and_same_augmentation_changes():
    scores = {
        "baseline_ce": report(0.60, 0.50),
        "augmentation_ce": report(0.65, 0.55),
        "weighted_ce": report(0.69, 0.60),
        "focal": report(0.67, 0.57),
        "label_smoothing": report(0.66, 0.58),
    }
    rows = runner.build_comparison(scores)
    weighted = next(row for row in rows if row["name"] == "weighted_ce")
    assert weighted["accuracy_vs_baseline"]["relative_percent_gain"] == pytest.approx(15)
    assert weighted["accuracy_loss_effect_same_augmentation"][
        "percentage_point_difference"
    ] == pytest.approx(4)
    assert "accuracy_loss_effect_same_augmentation" not in rows[1]


def test_demo_requires_explicit_opt_in(tmp_path):
    data = prepared_data(tmp_path, is_demo=True)
    with pytest.raises(ValueError, match="--allow-demo"):
        runner.preflight(data)
    assert runner.preflight(data, allow_demo=True)["is_demo"] is True


def test_preflight_requires_all_splits_and_nonempty_rows(tmp_path):
    data = prepared_data(tmp_path)
    (data / "test.csv").unlink()
    with pytest.raises(ValueError, match="test.csv"):
        runner.preflight(data)
    (data / "test.csv").write_text("id,image_path\n")
    with pytest.raises(ValueError, match="empty"):
        runner.preflight(data)


def test_matched_commands_share_seed_init_and_training_hyperparameters(tmp_path):
    args = runner.parser().parse_args([])
    baseline = runner.train_command(args, runner.EXPERIMENTS[0], tmp_path / "a")
    focal = runner.train_command(args, runner.EXPERIMENTS[3], tmp_path / "b")
    assert baseline[0] == focal[0] == sys.executable
    for option in (
        "--seed",
        "--epochs",
        "--batch-size",
        "--image-size",
        "--lr",
        "--weight-decay",
        "--gamma",
        "--label-smoothing",
        "--num-workers",
        "--threads",
        "--selection-metric",
    ):
        assert baseline[baseline.index(option) + 1] == focal[focal.index(option) + 1]
    assert "--pretrained" in baseline and "--pretrained" in focal
    assert "--freeze-backbone" in baseline and "--freeze-backbone" in focal
    assert args.channels_last is False
    assert "--no-channels-last" in baseline and "--no-channels-last" in focal
    assert "--no-augment" in baseline and "--augment" in focal
    channels_last_args = runner.parser().parse_args(["--channels-last"])
    for experiment in runner.EXPERIMENTS:
        command = runner.train_command(channels_last_args, experiment, tmp_path / experiment.name)
        assert "--channels-last" in command and "--no-channels-last" not in command


def test_runner_evaluates_test_once_after_validation_choice(tmp_path, monkeypatch):
    data = prepared_data(tmp_path)
    output = tmp_path / "experiment_results"
    args = runner.parser().parse_args(
        [
            "--data-dir",
            str(data),
            "--output-root",
            str(output),
            "--no-pretrained",
            "--channels-last",
        ]
    )
    calls = []
    val_scores = {
        "baseline_ce": 0.3,
        "augmentation_ce": 0.4,
        "weighted_ce": 0.7,
        "focal": 0.5,
        "label_smoothing": 0.6,
    }

    def fake_run(command, *, cwd, check):
        assert cwd == runner.ROOT and check is True
        calls.append(command)
        destination = Path(command[command.index("--output-dir") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        if command[2] == "fashion_classifier.train":
            (destination / "best.pt").touch()
            (destination / "summary.json").write_text(json.dumps({"best_epoch": 1}))
            (destination / "config.json").write_text(json.dumps({"resolved_device": "cpu"}))
        else:
            split = command[command.index("--split") + 1]
            if split == "test":
                # The comparison decision must be on disk before test is observed.
                saved = json.loads((output / "comparison.json").read_text())
                assert saved["winner"] == "weighted_ce"
                assert "winner_test_metrics" not in saved
                assert destination.parent.name == "weighted_ce"
                result = report(0.1, 0.1, split="test")
            else:
                result = report(0.5, val_scores[destination.parent.name])
            (destination / "metrics.json").write_text(json.dumps(result))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    comparison_path = runner.run_experiments(args)
    evaluation_calls = [call for call in calls if call[2] == "fashion_classifier.evaluate"]
    splits = [call[call.index("--split") + 1] for call in evaluation_calls]
    assert splits == ["val"] * 5 + ["test"]
    comparison = json.loads(comparison_path.read_text())
    assert comparison["winner"] == "weighted_ce"
    assert comparison["winner_test_metrics"]["macro_f1"] == 0.1
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["configuration"]["channels_last"] is True
    assert len(manifest["commands"]) == 11
    assert set(manifest["resolved_devices"].values()) == {"cpu"}
    assert all(command["runtime_seconds"] >= 0 for command in manifest["commands"])


def test_failed_training_stops_and_records_failure(tmp_path, monkeypatch):
    data = prepared_data(tmp_path)
    output = tmp_path / "failed_results"
    args = runner.parser().parse_args(["--data-dir", str(data), "--output-root", str(output)])
    calls = []

    def fail(command, *, cwd, check):
        calls.append(command)
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner.run_experiments(args)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert len(calls) == 1
    assert manifest["status"] == "failed"
    assert manifest["commands"][0]["returncode"] == 2


def test_existing_results_are_preserved(tmp_path):
    data = prepared_data(tmp_path)
    output = tmp_path / "saved_run"
    output.mkdir()
    saved = output / "comparison.json"
    saved.write_text("existing results")
    args = runner.parser().parse_args(["--data-dir", str(data), "--output-root", str(output)])
    with pytest.raises(ValueError, match="existing files"):
        runner.run_experiments(args)
    assert saved.read_text() == "existing results"
