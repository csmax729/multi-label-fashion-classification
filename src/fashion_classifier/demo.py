"""Generate clearly identified synthetic images to exercise the complete workflow."""

import argparse
import csv
import json
import random
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from .data import TASKS


def _reuse_complete_demo(output_dir, samples, seed):
    metadata_path = output_dir / "demo_metadata.json"
    if not metadata_path.is_file():
        raise FileExistsError(
            "Output directory already contains files without a demo marker; use a new directory"
        )
    metadata = json.loads(metadata_path.read_text())
    required = {
        "dataset_kind": "demo",
        "demo": True,
        "source": "Locally generated synthetic patterned fashion images",
        "samples": samples,
        "seed": seed,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError(
            "Existing demo metadata does not match the requested samples/seed; use a new directory"
        )
    # Regenerate away from the user's files and compare the complete payload.
    # A partial or modified dataset is rejected, and no existing file is written.
    with tempfile.TemporaryDirectory(prefix="fashion-demo-verify-") as staging:
        expected = Path(staging)
        generate_demo(expected, samples=samples, seed=seed)
        paths = [Path("styles.csv"), *(Path("images") / f"{i + 1}.png" for i in range(samples))]
        image_files = set((output_dir / "images").glob("*.png"))
        if image_files != {output_dir / path for path in paths[1:]}:
            raise ValueError(
                "Existing demo image set is incomplete or contains unexpected images; use a new directory"
            )
        for relative in paths:
            actual = output_dir / relative
            if not actual.is_file() or actual.read_bytes() != (expected / relative).read_bytes():
                raise ValueError(
                    f"Existing demo file was modified or is incomplete: {relative}; use a new directory"
                )
    return metadata


def generate_demo(output_dir, samples=40, seed=42):
    if samples < 16:
        raise ValueError("Use at least 16 samples to cover demo labels")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        return _reuse_complete_demo(output_dir, samples, seed)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    colours = (("Red", (175, 48, 55)), ("Blue", (45, 95, 175)), ("Green", (55, 145, 85)))
    for i in range(samples):
        gender = ("Men", "Women")[i % 2]
        article = ("Tshirts", "Shoes", "Handbags")[(i // 2) % 3]
        colour, rgb = colours[(i // 3) % 3]
        season = ("Summer", "Winter", "Spring", "Fall")[(i // 5) % 4]
        image = Image.new("RGB", (128, 128), (235, 235, 230))
        draw = ImageDraw.Draw(image)
        offset = rng.randint(-4, 4)
        if article == "Tshirts":
            draw.polygon(
                [
                    (37, 30),
                    (20, 48),
                    (35, 63),
                    (43, 55),
                    (43, 104),
                    (88, 104),
                    (88, 55),
                    (96, 63),
                    (110, 48),
                    (92, 30),
                ],
                fill=rgb,
            )
        elif article == "Shoes":
            draw.rounded_rectangle((20, 60 + offset, 111, 93 + offset), radius=15, fill=rgb)
            draw.rectangle((20, 90 + offset, 111, 96 + offset), fill="white")
        else:
            draw.rounded_rectangle((34, 48, 98, 105), radius=10, fill=rgb)
            draw.arc((47, 22, 84, 70), 180, 360, fill=rgb, width=7)
        if gender == "Men":
            draw.rectangle((8, 8, 19, 19), fill=(30, 30, 30))
        else:
            draw.ellipse((8, 8, 19, 19), fill=(30, 30, 30))
        for border in range(("Summer", "Winter", "Spring", "Fall").index(season) + 1):
            y = 113 + border * 3
            draw.line((30, y, 95, y), fill=(40, 40, 40), width=1)
        image.save(images_dir / f"{i + 1}.png")
        rows.append(
            {
                "id": str(i + 1),
                "gender": gender,
                "articleType": article,
                "baseColour": colour,
                "season": season,
            }
        )
    with (output_dir / "styles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", *TASKS])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "dataset_kind": "demo",
        "demo": True,
        "source": "Locally generated synthetic patterned fashion images",
        "samples": samples,
        "seed": seed,
        "limitations": "Artificial label cues; validates code execution only. Accuracy and latency do not substantiate real fashion-project resume claims.",
    }
    (output_dir / "demo_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/demo", type=Path)
    parser.add_argument("--samples", default=40, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    print(json.dumps(generate_demo(args.output_dir, args.samples, args.seed), indent=2))


if __name__ == "__main__":
    main()
