#!/usr/bin/env bash
set -euo pipefail

task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_repo_root"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/workspace/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
task_python="$task_repo_root/.venv/bin/python"
if [[ ! -x "$task_python" ]]; then
  printf '%s\n' 'Run uv sync --frozen before starting the demo.' >&2
  exit 1
fi

"$task_python" -m fashion_classifier.demo --output-dir data/demo --samples 40
"$task_python" -m fashion_classifier.data \
  --styles data/demo/styles.csv --images-dir data/demo/images \
  --output-dir data/demo_processed --seed 42
"$task_python" -m fashion_classifier.eda \
  --data-dir data/demo_processed --output-dir reports/demo_eda
"$task_python" -m fashion_classifier.train \
  --data-dir data/demo_processed --output-dir artifacts/demo \
  --epochs 2 --batch-size 8 --image-size 64 --device cpu --threads 2 \
  --loss weighted_ce --no-pretrained
"$task_python" -m fashion_classifier.evaluate \
  --data-dir data/demo_processed --checkpoint artifacts/demo/best.pt \
  --output-dir reports/demo_evaluation --batch-size 8 --device cpu --threads 2
"$task_python" -m fashion_classifier.benchmark \
  --checkpoint artifacts/demo/best.pt --image data/demo/images/1.png \
  --output reports/demo_latency.json --iterations 20 --warmup 3 --device cpu
printf '%s\n' 'Demo complete. Synthetic metrics verify execution only; they are not fashion accuracy.'
