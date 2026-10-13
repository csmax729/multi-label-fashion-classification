"""Measure batch-one image decoding + preprocessing + inference latency."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from fashion_classifier.inference import Predictor


def benchmark(predictor: Predictor, image_bytes: bytes, warmup: int, iterations: int) -> dict:
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    timings = []
    for iteration in range(warmup + iterations):
        start = time.perf_counter()
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            predictor.predict(image)
        elapsed = (time.perf_counter() - start) * 1000.0
        if iteration >= warmup:
            timings.append(elapsed)
    return {
        "device": str(predictor.device),
        "batch_size": 1,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "mean_ms": float(np.mean(timings)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
        "includes": [
            "image decoding",
            "preprocessing",
            "model inference",
            "softmax and label mapping",
        ],
        "excludes": ["model startup", "disk reads", "HTTP transport", "UI rendering"],
        "model_info": predictor.model_info,
        "meets_1200ms_p95_on_this_machine": float(np.percentile(timings, 95)) < 1200.0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations must be positive")
    predictor = Predictor(args.checkpoint, device=args.device)
    report = benchmark(predictor, args.image.read_bytes(), args.warmup, args.iterations)
    report["input_image"] = str(args.image.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
