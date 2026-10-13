# Fashion image classification

A working learning project for predicting **gender, article type, base colour, and season**
from a product image. A shared ResNet50 feeds four categorical output heads. Includes
data auditing, EDA, reproducible training, three imbalance-aware loss alternatives,
evaluation, latency measurement, FastAPI, and Streamlit.

This is often called multi-label classification on a résumé. More precisely, it is
**multi-task, multi-output classification**: each task has one mutually exclusive label.
We use softmax and cross-entropy per task, rather than independent sigmoid outputs.

**Evidence policy:** 82% accuracy, 15% improvement, 3:1 imbalance, and <1.2 s latency
are hypotheses until measured. Synthetic demo results never establish fashion accuracy.
Read [the interview walkthrough](docs/PROJECT_INTERVIEW_COACHING.md) and
[the implementation stages](docs/BUILD_STEPS.md) to explain the work honestly.
The [architecture diagrams](docs/ARCHITECTURE.md) show how training and serving fit together.

Download the complete interview coaching guide to your computer:

- [PDF](https://github.com/csmax729/multi-label-fashion-classification/raw/refs/heads/main/docs/Fashion_Project_Interview_Coaching.pdf)
- [Editable Word document](https://github.com/csmax729/multi-label-fashion-classification/raw/refs/heads/main/docs/Fashion_Project_Interview_Coaching.docx)
- [Markdown source](docs/PROJECT_INTERVIEW_COACHING.md)

The guide includes the full project walkthrough, four speaking versions, technical
practice questions, and exercises for explaining the project confidently.

The public dataset has now been audited and an initial model trained on all 31,106
training examples. The one-epoch, frozen-backbone, 96-pixel baseline measured **57.73%
validation mean task accuracy** and **58.16% test mean task accuracy**, with **10.20%
test exact match**. Warm local HTTP latency was **27 ms median / 163 ms p95**.
These are initial measurements; 82% accuracy and a 15% gain have not been established.
See [execution evidence](docs/VALIDATION.md) for per-task results and measurement scope.

## Install

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/) are required; this environment
uses Python 3.12 and CPU PyTorch. Dependencies are resolved in `uv.lock`.

```bash
cd /workspace/multi-label-fashion-classification
uv sync --frozen
source .venv/bin/activate
```

The locked default uses CPU PyTorch. GPU training needs a matching official PyTorch
CUDA environment; `--device cuda` alone cannot add GPU hardware or CUDA support.

## Run a complete demo

```bash
bash scripts/demo.sh
```

This creates patterned synthetic images, audits and splits them, produces EDA plots,
trains a real ResNet50 checkpoint on CPU, evaluates it, and measures inference latency.
It verifies the pipeline, not fashion model quality. All generated data, weights,
and reports are ignored by Git.

## Use the public fashion dataset

Source: [Kaggle Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small).
The source describes roughly 44,000 product images; our audit reports the actual rows
and usable images. The small dataset has low-resolution images, which is a modelling
limitation even when resized to 224×224.

```bash
python -m fashion_classifier.download --output-dir data/raw
```

The downloader verifies the archive, selects one canonical copy if Kaggle bundles
duplicate directory layouts, and records source provenance. Alternatively, download
and extract from Kaggle into `data/raw`. Locate `styles.csv` and `images/`, then run:

```bash
python -m fashion_classifier.data \
  --styles data/raw/styles.csv --images-dir data/raw/images \
  --output-dir data/processed --seed 42
python -m fashion_classifier.eda --data-dir data/processed --output-dir reports/eda
```

For cloud downloads, allow `www.kaggle.com` and `storage.googleapis.com` in environment
settings. The downloader preserves TLS verification and never asks for credentials
in chat. ImageNet weights download separately from `download.pytorch.org`.

## Train and evaluate

```bash
python -m fashion_classifier.train \
  --data-dir data/processed --output-dir artifacts/fashion \
  --epochs 10 --batch-size 32 --image-size 224 --loss weighted_ce
python -m fashion_classifier.evaluate \
  --data-dir data/processed --checkpoint artifacts/fashion/best.pt \
  --output-dir reports/evaluation
```

Training starts with ImageNet weights and a frozen backbone by default. This learns
the new output heads quickly. Inspect `train --help` to unfreeze the backbone and use
a smaller learning rate for fine-tuning. Use `--no-pretrained` only for an offline
smoke run or an explicit from-scratch baseline. Validation selects checkpoints;
the held-out test split estimates final generalization.

For this CPU machine, the initial full-data run uses 96×96 inputs, batch size 64,
four CPU threads, and `--channels-last` to improve convolution throughput. This is
an initial baseline with a frozen backbone, not a claim that one epoch reaches the
résumé's accuracy. The generic commands above use 224-pixel inputs and need more compute.

Loss choices are `ce` (baseline), `weighted_ce`, `focal`, and `label_smoothing`.
They are alternatives to compare, not three losses to stack indiscriminately.
Class weights and label vocabularies are derived from the training split.

```bash
python scripts/run_experiments.py --data-dir data/processed --epochs 10
python -m fashion_classifier.benchmark --help
```

See [the experiment guide](docs/EXPERIMENTS.md) for a faster CPU recipe and how
to distinguish relative improvement from percentage points.

The experiment runner compares baseline, augmentation, and loss variants on the
same splits and seed. It selects the winner on validation macro-F1 before testing.
Accuracy is reported per task, averaged over tasks, and as exact match across all
four labels; those are different metrics. Macro-F1 helps expose poor rare-class
performance hidden by aggregate accuracy.

## Serve predictions

In two terminals with the virtual environment activated:

```bash
FASHION_CHECKPOINT=artifacts/demo/best.pt python -m uvicorn \
  fashion_classifier.api:app --host 0.0.0.0 --port 8000
```

```bash
FASHION_API_URL=http://127.0.0.1:8000 streamlit run \
  src/fashion_classifier/ui.py --server.headless true --server.port 8501
```

`GET /health` reports model readiness. `POST /predict` accepts a multipart image
field named `file`. To use a real fashion model, set `FASHION_CHECKPOINT` to its
checkpoint and restart the API. The UI marks synthetic checkpoints as demonstrations.

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail -F file=@data/demo/images/1.png http://127.0.0.1:8000/predict
```

The Dockerfile packages the API; mount your local trained checkpoint into the
container. No public deployment has been performed by creating these serving files.

Measure the full warm HTTP request path while the API is running:

```bash
python scripts/benchmark_api.py --image data/demo/images/1.png
```

Use a representative real image and the fashion checkpoint for real serving results.
See [container instructions](docs/DEPLOYMENT.md) for the packaged API.

## Verify

```bash
python -m pytest
```

See [validation evidence](docs/VALIDATION.md) for checks actually run in this environment
and remaining real-data experiments. No data or trained weights are committed.
