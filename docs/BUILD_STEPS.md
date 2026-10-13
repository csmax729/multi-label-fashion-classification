# What we built, step by step

This log explains the implementation stages. `VALIDATION.md` records the execution
evidence separately so that code availability is not confused with measured quality.

## 1. Define the problem

Input: one product image. Outputs: gender, article type, base colour, and season.
Each output chooses one class. Four heads share the expensive visual representation.

Interview explanation: “I used multi-task learning because the same garment shape,
colour, and texture features can help several attributes. The shared backbone makes
four predictions in one pass.” Gender and season are dataset annotations, not facts
an image can always determine, which limits the interpretation of predictions.

## 2. Inspect and clean the dataset

The input is Kaggle's `styles.csv` plus images named by product ID. The preparation
command checks CSV parsing, missing labels, duplicate IDs, missing and unreadable
images, and duplicate image content. Every exclusion or fallback belongs in the audit.

Interview explanation: “I checked whether each metadata row had a usable image and
valid labels. I recorded rejected rows rather than quietly accepting corrupt inputs.”
Cleaning means fixing input usability; it does not establish an accuracy improvement.

## 3. Separate training, validation, and test data

The split is deterministic. Duplicate image content must stay together so the same
visual input does not appear in both training and evaluation. The training split owns
class mappings and class weights. Evaluation retains and reports unsupported labels.

Interview explanation: “Validation guided model selection. Test data stayed untouched
until the final selection. I prevented duplicate images from inflating my metrics.”
Product variants may still be correlated without exact duplicate pixels; a stronger
future split would use reliable product-family identifiers if the source provides them.

## 4. Perform EDA

The EDA command creates counts, per-task distributions, imbalance ratios, and plots
from the prepared manifests. Interpret maximum/minimum class ratio with class counts;
a rare singleton can make the ratio huge. Do not assume the résumé's 3:1 figure.

Interview explanation: “I used class distributions to decide whether weighted losses
were worth testing, and macro-F1 to check if they actually helped minority classes.”

The real audit completed in this environment: 44,446 metadata rows yielded 44,420
usable examples. Twenty-one rows had missing task labels and five images were missing.
Twenty-two display-name rows needed CSV comma repair. The split sizes are 31,106
training, 6,657 validation, and 6,657 test images.

| Task | Classes | Largest/smallest class ratio |
| --- | ---: | ---: |
| Gender | 5 | 33.82:1 |
| Article type | 142 | 7,068:1 |
| Base colour | 47 | 1,945.4:1 |
| Season | 4 | 7.20:1 |

There are 653 duplicate pixel-content groups containing 780 repeated rows. Sixty-eight
groups have conflicting labels: they stay together in one split and are flagged for
review. Six singleton article classes must remain in training; twelve article classes
have no examples in each evaluation split. This limits what held-out metrics can say
about those classes. The original “3:1” claim does not describe this dataset's full
class distributions.

## 5. Preprocess and augment

Validation and inference use the same deterministic resize and ImageNet normalization.
Training adds random image transformations. Inspect `data.get_transforms` for the
implemented choices and parameters. Keep colour changes modest when predicting colour;
aggressive hue changes can make augmentation labels wrong.

Interview explanation: “Augmentation changes training views while retaining the label.
It does not add independent products or solve every rare-class problem.”

## 6. Build ResNet50 with four heads

ResNet50 has residual connections that help gradients pass through a deep network.
Its final ImageNet classifier is replaced with an identity mapping. The resulting
2,048-dimensional feature vector feeds four task-specific linear classifiers.

ImageNet pretraining supplies a useful starting representation. First train the heads
with the backbone frozen; fine-tune selected backbone layers or the full backbone with
a smaller learning rate when the dataset and compute permit it.

## 7. Compare loss functions

Weighted cross-entropy gives greater influence to underrepresented training classes.
Focal loss reduces contributions from easy examples. Label smoothing prevents targets
from demanding absolute certainty. The multi-task objective averages losses across
heads, avoiding automatic domination solely because one head has more classes.

Interview explanation: “I compared loss variants against ordinary cross-entropy on
identical splits. I did not choose a loss just because it was more complicated.”

## 8. Train and select a checkpoint

Training records configuration, provenance, label ordering, epoch metrics, and a best
validation checkpoint. Seeds make runs easier to compare; they do not guarantee every
accelerator computation is bit-for-bit deterministic. Saved model weights need the
same label mapping and preprocessing to produce meaningful predictions.

## 9. Evaluate and run ablations

Evaluation reports per-task accuracy, macro-F1, class support, confusion matrices,
mean task accuracy, and exact-match accuracy. The experiment runner holds split and
training settings constant and compares augmentation separately from loss changes.

Interview explanation: “An average task accuracy of 82% would not mean 82% of images
had all four labels correct. I report exact match separately.”

For a baseline accuracy of 70% and a new accuracy of 80.5%, the improvement is
**10.5 percentage points** or **15% relative**. These numbers are an explanatory
example, not project results. A result must specify which metric, split, and baseline.

## 10. Serve and measure

FastAPI loads a checkpoint once and validates uploads before inference. Streamlit
provides an upload and prediction interface. The benchmark measures warm batch-one
inference; API requests additionally include HTTP/upload overhead. Record p50 and p95,
hardware, image size, cold-start behaviour, and concurrency before claiming latency.

Interview explanation: “I measured the complete serving path on specified hardware.
I separated model timing from HTTP latency and kept the model loaded between requests.”

## 11. Preserve reproducibility

`pyproject.toml` describes dependencies and `uv.lock` records exact versions. Generated
datasets, reports, and checkpoints stay out of Git. Future cloud tasks use the existing
checkout, install from the lockfile, and restart services explicitly.

## 12. Distinguish what is complete

Implementation and synthetic smoke validation can finish without a downloaded fashion
dataset. Real EDA, a trained fashion checkpoint, generalization results, and any
performance-improvement claim require the actual public dataset and experiments.
