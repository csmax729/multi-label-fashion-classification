# Execution evidence

Validation performed on 13 October 2026 in this cloud environment, using Python
3.12.14, PyTorch 2.8.0 CPU, torchvision 0.23.0 CPU, four CPU cores of quota, and no GPU.

## Data

The real public Kaggle dataset was downloaded over verified HTTPS. Its archive digest
is `e223d09ba5fb5829892e507989dba11b2549a31fabed994ad5353e4b6ef87b7a`.
This locally recorded digest supplies provenance; it is not a publisher checksum.
ZIP CRCs and the recovered extracted files were checked. Two duplicate directory
layouts were verified equivalent, and only the canonical root copy was prepared.

- Metadata rows: 44,446; usable rows: 44,420.
- Exclusions: 21 missing task labels, 5 missing images.
- CSV repairs: 22 final display-name comma rows, reported in the audit.
- Split sizes: 31,106 training, 6,657 validation, 6,657 test.
- Duplicate-content groups: 653; repeated rows beyond unique images: 780.
- Conflicting-label duplicate groups: 68, retained together and flagged.
- Training classes: gender 5, article type 142, colour 47, season 4.
- No unknown evaluation labels; absent evaluation classes are explicitly reported.

Generated evidence: `data/processed/quality_report.json`, `data/processed/manifest.json`,
and `reports/eda/eda_report.json` plus four inspected class-distribution plots.
The measured class ratios are in `BUILD_STEPS.md`; a general “3:1” claim is unsupported.

## Automated and functional checks

- `uv sync --frozen`: passed, including repeat installation from the lockfile.
- `python -m pytest -q`: 66 tests passed. One dependency deprecation warning; no failures.
- Ruff lint and format checks: passed.
- `bash scripts/demo.sh`: passed twice, including safe reuse of generated demo data.
- ImageNet V2 download and a 224×224 ResNet50 forward pass: passed, finite outputs.
- Synthetic training, checkpoint reload, evaluation, and inference: passed.
- Synthetic checkpoint initialization followed by unfrozen fine-tuning: passed.
- Channels-last memory layout: output-equivalence and checkpoint round-trip passed.
- Five matched synthetic ablations: passed; selection used validation macro-F1;
  only the selected winner was evaluated on the test split.
- Live FastAPI `/health` and multipart `/predict`: passed with a real model artifact.
- Live Streamlit health and AppTest rendering against that API: passed.
- API/UI tests cover successful uploads, missing checkpoints, malformed/oversized
  images, multipart truncation, demo provenance, and four rendered prediction cards.
- Final-source Docker image built and served a synthetic checkpoint: `/health` and
  `/predict` returned HTTP 200, with nonroot execution and a read-only weights mount.
  The temporary container was removed; the local image remains. Build reused existing
  proxy bindings and an ephemeral trusted CA bundle, keeping TLS verification enabled.
  See `reports/docker_smoke.json` and `DEPLOYMENT.md`. No public service was created.

The synthetic 40-image dataset splits into 28/6/6. Its metrics demonstrate execution
only. A standalone 224-pixel ImageNet-initialized head-training smoke also passed on
those synthetic images. These runs do not establish fashion accuracy or improvement.

## Timing boundaries

`fashion_classifier.benchmark` includes image decoding, preprocessing, model inference,
and label mapping. `scripts/benchmark_api.py` additionally measures the HTTP upload and
response. Both measure warm batch-one requests, excluding model loading and UI rendering.

The initial synthetic 64-pixel HTTP run measured p50 14.9 ms and p95 23.1 ms over
30 requests after five warmups. This is a demo checkpoint on an otherwise lightly
loaded local machine, not a real-fashion deployment claim. Background training increased
tail latency in a later demo run, so performance should be measured without contention.

## Initial real-fashion training

A full pass over all 31,106 training examples completed with ImageNet V2 initialization,
a frozen backbone, weighted cross-entropy, six training augmentation operators,
96×96 inputs, batch size 64, four CPU threads, seed 42, and channels-last storage.
The epoch including validation took 690.1 seconds. `artifacts/fashion/best.pt`
contains the trained checkpoint and all four class vocabularies.

Validation on 6,657 images: mean task accuracy **57.73%**, mean task macro-F1
**0.4458**, and all-four exact match **9.73%**. The checkpoint was selected using
validation macro-F1; no test examples were used for fitting or selection.

This is a one-epoch head-training baseline, not a fully tuned model. It establishes
that the complete real-data training workflow runs and provides an honest starting
measurement. An 82% validation-accuracy claim is not supported by this run.

The held-out test evaluation executed on all 6,657 test images and passed checkpoint
vocabulary and dataset-provenance checks. Aggregate mean task accuracy is **58.16%**,
mean task macro-F1 **0.4427**, and all-four exact match **10.20%**.

| Task | Test accuracy | Macro-F1 | Test support |
| --- | ---: | ---: | ---: |
| Gender | 76.03% | 0.6058 | 6,657 |
| Article type | 63.26% | 0.3726 | 6,657 |
| Base colour | 35.54% | 0.1842 | 6,657 |
| Season | 57.82% | 0.6082 | 6,657 |

The API was restarted with this real checkpoint. `/health`, a real-image multipart
`/predict`, and the live Streamlit rendering check passed. The UI no longer shows a
synthetic-demo warning. Evidence is in `reports/fashion_api_smoke.json`.

After training and evaluation stopped competing for CPU, the real checkpoint's warm
HTTP benchmark measured **p50 27.1 ms / p95 163.0 ms**, over 30 sequential requests
after five warmups. Decode/preprocess/inference alone in a separate run measured
**p50 55.1 ms / p95 242.6 ms**. Both runs use one real JPEG and 96-pixel model inputs
on this CPU machine. Different runtime conditions produce timing variation; these
results do not cover cold starts, concurrency, public network latency, or UI rendering.
Reports: `reports/fashion_api_latency.json` and `reports/fashion_latency.json`.

The entire real-data ablation suite has not been run. The synthetic five-configuration
smoke confirms orchestration, not a 15% fashion accuracy gain. A stronger model needs
further head training and backbone fine-tuning guided by validation; colour performance
and rare-class support deserve particular attention.

## Remaining scientific limits

Exact duplicate grouping does not detect perceptual duplicates or related product
families. Sixty-eight duplicate groups have conflicting annotations. Twelve article
classes are absent from each held-out split, and six singleton article classes must
remain training-only. One seed cannot establish statistical confidence in gains.
Source image mutations are not checked by checkpoint CSV hashes alone.

Full-training accuracy, improvement against a matched baseline, robustness across
independent datasets, and public deployment must be supported by their own results.
No 82% accuracy or 15% gain is assumed or hard-coded in this project.
