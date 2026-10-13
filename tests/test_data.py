import csv
import json
import os
import subprocess
import sys
import zipfile

import pytest
from PIL import Image

from fashion_classifier.data import TASKS, prepare_data
from fashion_classifier.demo import generate_demo
from fashion_classifier.download import download_dataset, safe_extract


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_deterministic_group_split_and_train_only_vocabulary(tmp_path):
    raw = tmp_path / "raw"
    generate_demo(raw, samples=64)
    # Different IDs with identical content must stay in the same split.
    with Image.open(raw / "images" / "1.png") as source:
        source.save(raw / "images" / "65.png", compress_level=0)
    rows = read_rows(raw / "styles.csv")
    rows.append({**rows[0], "id": "65"})
    # A label occurring in one content group must be reserved for training.
    rows[1]["season"] = "RareSeason"
    with (raw / "styles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", *TASKS])
        writer.writeheader()
        writer.writerows(rows)
    report = prepare_data(raw / "styles.csv", raw / "images", tmp_path / "first")
    prepare_data(raw / "styles.csv", raw / "images", tmp_path / "second")
    split_rows = {
        split: read_rows(tmp_path / "first" / f"{split}.csv") for split in ("train", "val", "test")
    }
    membership = {row["id"]: split for split, records in split_rows.items() for row in records}
    assert membership["1"] == membership["65"]
    assert membership["2"] == "train"
    assert report["duplicate_content_groups"] >= 1
    assert report["provenance"]["dataset_kind"] == "demo"
    labels = json.loads((tmp_path / "first" / "labels.json").read_text())
    assert labels == {task: sorted({row[task] for row in split_rows["train"]}) for task in TASKS}
    assert all(
        not unknown
        for per_task in report["unknown_labels"].values()
        for unknown in per_task.values()
    )
    for split in split_rows:
        assert (tmp_path / "first" / f"{split}.csv").read_bytes() == (
            tmp_path / "second" / f"{split}.csv"
        ).read_bytes()
    assert sum(report["split_counts"].values()) == 65


def test_preparation_accounts_for_bad_rows_and_missing_images(tmp_path):
    raw = tmp_path / "raw"
    generate_demo(raw, samples=40)
    with (raw / "styles.csv").open("a") as handle:
        handle.write("999,Men,Tshirts,Red,Summer\n")
        handle.write("1000,Men,Tshirts,,Summer\n")
        handle.write("1001,Men\n")
    report = prepare_data(raw / "styles.csv", raw / "images", tmp_path / "processed")
    assert report["source_rows"] == 43
    assert report["excluded"]["missing_image"] == 1
    assert report["excluded"]["missing_task_label"] == 1
    assert report["excluded"]["malformed_csv"] == 1
    assert report["source_rows"] == report["usable_rows"] + sum(report["excluded"].values())


def test_zip_rejects_path_traversal_before_extracting(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("ordinary.txt", "ok")
        handle.writestr("../escape.txt", "bad")
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, destination)
    assert not destination.exists()
    assert not (tmp_path / "escape.txt").exists()


def test_repaired_display_names_corrupt_images_and_conflicting_ids_are_audited(tmp_path):
    raw = tmp_path / "raw"
    generate_demo(raw, samples=40)
    rows = read_rows(raw / "styles.csv")
    with (raw / "styles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", *TASKS, "productDisplayName"])
        writer.writeheader()
        writer.writerows({**row, "productDisplayName": "ordinary"} for row in rows)
        # This last field contains an unquoted comma, as in the public dataset.
        handle.write("41,Men,Tshirts,Red,Summer,Name,with comma\n")
        writer.writerow(
            {**rows[0], "gender": "ConflictingGender", "productDisplayName": "conflict"}
        )
    (raw / "images" / "41.png").write_bytes(b"not an image")
    report = prepare_data(raw / "styles.csv", raw / "images", tmp_path / "processed")
    assert report["repaired_display_name_rows"] == 1
    assert report["excluded"]["invalid_image"] == 1
    assert report["excluded"]["conflicting_duplicate_id"] == 2
    assert report["source_rows"] == report["usable_rows"] + sum(report["excluded"].values())


def test_demo_reuse_preserves_user_files_and_rejects_modified_or_unrelated_data(tmp_path):
    raw = tmp_path / "demo"
    first = generate_demo(raw, samples=40)
    preserved = raw / "my_notes.txt"
    preserved.write_text("Interview notes")
    csv_before = (raw / "styles.csv").read_bytes()
    assert generate_demo(raw, samples=40) == first
    assert preserved.read_text() == "Interview notes"
    assert (raw / "styles.csv").read_bytes() == csv_before
    with pytest.raises(ValueError, match="metadata"):
        generate_demo(raw, samples=41)
    (raw / "images" / "1.png").write_bytes(b"user changed this image")
    with pytest.raises(ValueError, match="modified"):
        generate_demo(raw, samples=40)
    assert (raw / "images" / "1.png").read_bytes() == b"user changed this image"
    unknown = tmp_path / "unrelated"
    unknown.mkdir()
    (unknown / "1.png").write_text("User file")
    with pytest.raises(FileExistsError, match="without a demo marker"):
        generate_demo(unknown)
    assert (unknown / "1.png").read_text() == "User file"


def test_cached_download_recovers_verified_duplicate_layout_without_overwriting(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    csv_bytes = b"id,gender,articleType,baseColour,season\n1,Men,Tshirts,Red,Summer\n"
    with zipfile.ZipFile(raw / "dataset.zip", "w") as handle:
        for prefix in ("", "myntradataset/"):
            handle.writestr(f"{prefix}styles.csv", csv_bytes)
            handle.writestr(f"{prefix}images/1.png", b"image fixture")
    safe_extract(raw / "dataset.zip", raw)
    metadata = download_dataset(raw)
    assert metadata["layout"]["selected_layout"] == "."
    assert metadata["layout"]["ignored_duplicate_layouts"] == ["myntradataset"]
    assert metadata["extraction"] == {"verified_existing_files": 4, "extracted_new_files": 0}
    assert metadata["styles"] == str((raw / "styles.csv").resolve())
    assert download_dataset(raw)["archive_sha256"] == metadata["archive_sha256"]
    (raw / "images" / "1.png").write_bytes(b"modified file")
    with pytest.raises(FileExistsError, match="checksum differs"):
        download_dataset(raw)
    assert (raw / "images" / "1.png").read_bytes() == b"modified file"


def test_split_is_identical_across_python_hash_seeds(tmp_path):
    raw = tmp_path / "raw"
    generate_demo(raw, samples=64)
    for seed in ("1", "2"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "fashion_classifier.data",
                "--styles",
                str(raw / "styles.csv"),
                "--images-dir",
                str(raw / "images"),
                "--output-dir",
                str(tmp_path / seed),
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
    for split in ("train", "val", "test"):
        assert (tmp_path / "1" / f"{split}.csv").read_bytes() == (
            tmp_path / "2" / f"{split}.csv"
        ).read_bytes()
