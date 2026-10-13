# Measuring the effects of augmentation and loss functions

The experiment runner compares five configurations on one prepared dataset. It
uses the same split files, seed, ImageNet ResNet50 initialization, optimizer,
epoch budget, batch size, image size, and backbone-freezing setting for each run.
The default seed is 42 and the default backbone is frozen. Each process starts
with a fresh optimizer; a previous experiment's trained weights are never reused.

| Configuration | Loss | Training augmentation | What to compare |
| --- | --- | --- | --- |
| `baseline_ce` | Unweighted cross entropy | Disabled | Reference |
| `augmentation_ce` | Unweighted cross entropy | Enabled | Compare with baseline to isolate augmentation |
| `weighted_ce` | Class-balanced cross entropy | Enabled | Compare with augmented CE to isolate the loss change |
| `focal` | Class-balanced focal loss, gamma 2 | Enabled | Compare with augmented CE to isolate the loss change |
| `label_smoothing` | Class-balanced smoothed cross entropy, epsilon 0.1 | Enabled | Compare with augmented CE to isolate the loss change |

Comparing a custom-loss run with `baseline_ce` measures the combined changes to
augmentation and loss. It does not establish how much each change contributes.
Class weights come from the training labels only. Validation and test inputs do
not receive training augmentation.

## Run a comparison

Activate the project environment and prepare the real dataset as described in
the README. Run from the repository root. The standard defaults use 224-pixel
images, batch size 32, three epochs, pretrained weights, and a frozen backbone:

```bash
python scripts/run_experiments.py \
  --data-dir data/processed \
  --output-root artifacts/experiments-standard
```

On the current cloud CPU environment, four threads and smaller inputs are a
practical first budget. There is no GPU in this instance. Decide the epoch
budget before inspecting the final test results:

```bash
python scripts/run_experiments.py \
  --data-dir data/processed \
  --output-root artifacts/experiments-cpu96 \
  --device cpu --threads 4 --image-size 96 --batch-size 64 \
  --epochs 1 --channels-last
```

This still runs five full training jobs, validation at each epoch, validation
evaluation for each selected checkpoint, and one final held-out test evaluation.
Runtime depends on the CPU, storage, image decoding, and selected epoch budget;
use the recorded runtimes to plan subsequent work. A 96-pixel, one-epoch result
is a distinct training configuration and cannot substantiate a claim about a
224-pixel model trained for more epochs.

`--channels-last` changes convolution memory storage and can improve CPU
throughput. The runner applies the same setting to all five training jobs and
records it in the manifest and training configurations. Its default is disabled.
For GPU training, choose a supported device and measure an appropriate batch
size; do not assume this CPU recipe is optimal for another machine.

## Read the evidence

Each configuration has its checkpoint, training configuration, class counts,
epoch history, validation metrics, and confusion matrices. `run_manifest.json`
records the dataset hashes, source, requested and resolved devices, seed,
hyperparameters, exact commands, return codes, and runtime. `comparison.json`
reports mean task accuracy and macro F1 plus changes against both reference
configurations. Existing results are preserved: choose a new output root for a
new run.

The winner is selected using **validation macro F1**. Its validation decision is
saved before its test evaluation runs. Test results do not select the loss,
checkpoint, or configuration. Keep the held-out test split out of subsequent
tuning decisions.

Relative percentage gain and percentage-point difference are different. An
accuracy increase from 60% to 69% is a **15% relative gain** and a **9 percentage
point difference**. A zero baseline has no defined relative gain. Negative
changes are reported rather than hidden. A single seed cannot establish
statistical significance; stronger evidence needs a predeclared multi-seed
validation protocol and one final test evaluation.

Demo data requires explicit `--allow-demo`. Its measurements validate execution,
not real fashion classification quality. The 82% accuracy and roughly 15%
improvement in the proposed resume remain unproven until the corresponding
real-data experiment reports support them.
