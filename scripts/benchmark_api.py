"""Measure warm, sequential HTTP prediction latency against a running API."""

from __future__ import annotations

import argparse
import json
import mimetypes
import platform
import time
from pathlib import Path

import numpy as np
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("reports/api_latency.json"))
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive; warmup must be nonnegative")
    endpoint = args.api_url.rstrip("/")
    content = args.image.read_bytes()
    mime_type = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
    durations = []
    with requests.Session() as session:
        health = session.get(f"{endpoint}/health", timeout=10)
        health.raise_for_status()
        model_info = health.json()["model_info"]
        for iteration in range(args.warmup + args.iterations):
            start = time.perf_counter()
            response = session.post(
                f"{endpoint}/predict",
                files={"file": (args.image.name, content, mime_type)},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            if set(result["predictions"]) != {"gender", "articleType", "baseColour", "season"}:
                raise ValueError("API did not return all four classification tasks")
            duration_ms = (time.perf_counter() - start) * 1000
            if iteration >= args.warmup:
                durations.append(duration_ms)
    report = {
        "api_url": endpoint,
        "client_machine": platform.machine(),
        "warmup_requests": args.warmup,
        "measured_requests": args.iterations,
        "concurrency": 1,
        "image_bytes": len(content),
        "p50_ms": float(np.percentile(durations, 50)),
        "p95_ms": float(np.percentile(durations, 95)),
        "mean_ms": float(np.mean(durations)),
        "max_ms": max(durations),
        "includes": ["HTTP upload", "image decoding", "preprocessing", "inference", "response"],
        "excludes": ["model loading", "Streamlit rendering", "concurrent-request queuing"],
        "model_info": model_info,
        "limitation": "Warm sequential measurements for this checkpoint, image, and machine only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
