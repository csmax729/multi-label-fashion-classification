"""Produce inspectable class counts, imbalance statistics and static plots."""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

from .data import TASKS


def analyze(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".matplotlib").resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str((output_dir / ".cache").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_by_split = {}
    for split in ("train", "val", "test"):
        with (data_dir / f"{split}.csv").open(newline="") as handle:
            rows_by_split[split] = list(csv.DictReader(handle))
    rows_by_split["all"] = [row for rows in rows_by_split.values() for row in rows]
    report = {
        "source": str(data_dir.resolve()),
        "provenance": json.loads((data_dir / "manifest.json").read_text()),
        "splits": {},
        "note": "Imbalance is max/min nonzero count per attribute, not a joint-label ratio. Plot shows up to the 20 most frequent classes; JSON includes every class.",
    }
    for split, rows in rows_by_split.items():
        report["splits"][split] = {"samples": len(rows), "tasks": {}}
        for task in TASKS:
            counts = Counter(row[task] for row in rows)
            values = list(counts.values())
            report["splits"][split]["tasks"][task] = {
                "counts": dict(sorted(counts.items())),
                "classes": len(counts),
                "max_to_min_ratio": max(values) / min(values) if values else None,
            }
    for task in TASKS:
        counts = Counter(row[task] for row in rows_by_split["all"])
        top = counts.most_common(20)[::-1]
        figure, axes = plt.subplots(figsize=(9, max(3, len(top) * 0.3)))
        axes.barh([label for label, _ in top], [count for _, count in top], color="#356d9a")
        axes.set_xlabel("Validated product images")
        kind = report["provenance"]["dataset_kind"]
        axes.set_title(f"{task}: class distribution ({kind} dataset)")
        figure.tight_layout()
        figure.savefig(output_dir / f"{task}_distribution.png", dpi=150)
        plt.close(figure)
    (output_dir / "eda_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed", type=Path)
    parser.add_argument("--output-dir", default="reports/eda", type=Path)
    args = parser.parse_args()
    report = analyze(args.data_dir, args.output_dir)
    print(
        json.dumps(
            {
                "samples": report["splits"]["all"]["samples"],
                "dataset_kind": report["provenance"]["dataset_kind"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
