# Interview preparation

## A 60-second explanation

“I built an image classification pipeline for four product attributes: gender,
article type, base colour, and season. It audits metadata and images, creates
deterministic splits, and trains a shared ResNet50 with four output heads. I implemented
weighted cross-entropy, focal loss, and label smoothing as alternatives, plus training
augmentation. Evaluation includes per-task accuracy, macro-F1, and exact match. A
FastAPI backend and Streamlit interface serve saved checkpoints. I keep measured
results separate from synthetic pipeline tests.”

Replace the last sentence with specific real-data results only after the recorded
experiments support them. Do not state “deployed” if you only ran services locally.

## Questions you should be able to answer

1. **Why four heads rather than four independent models?** Shared features reduce
   inference cost and parameters; unrelated or conflicting tasks can still interfere.
2. **Is this multi-label or multi-class?** It is four multi-class tasks for one image.
   Each head uses softmax; sigmoid would suit independent yes/no labels instead.
3. **Why ResNet50?** It is a well-understood transfer-learning baseline with pretrained
   weights. Compare a smaller network if latency or memory matter more than capacity.
4. **What do residual connections do?** They add a shortcut around a learned block,
   making it easier to optimize deep models without requiring every block to transform
   the representation strongly.
5. **Why ImageNet normalization?** It matches the distribution expected by pretrained
   weights. Train, validation, and inference preprocessing must agree.
6. **What does freezing mean?** Backbone parameters do not receive optimizer updates.
   Frozen batch-normalization statistics should also remain fixed during head training.
7. **When would you fine-tune?** After learning heads, use a smaller learning rate to
   adapt the visual features, monitoring validation performance for overfitting.
8. **How did you prevent leakage?** Keep duplicate image content in one split and learn
   mappings/weights from training only. Exact hashing does not detect near duplicates.
9. **What did EDA find?** Answer using `reports/eda`, naming actual sample counts,
   missingness, and class ratios: 44,420 usable images; task imbalances range from
   7.20:1 for season to 7,068:1 for article type. These are measured audit results.
10. **How did you handle rare classes?** Audit support, ensure training vocabulary
    coverage, compare imbalance-aware losses, and inspect macro-F1. Classes absent from
    evaluation cannot support a performance claim.
11. **How does weighted cross-entropy work?** It multiplies each example's negative
    log probability by a class weight derived from training counts.
12. **How does focal loss work?** It multiplies the cross-entropy contribution by
    `(1 - p_t)^gamma`, reducing the influence of confidently correct examples.
13. **What does label smoothing change?** It spreads some target probability across
    other classes, discouraging overconfident fits. It does not inherently rebalance data.
14. **Did you use all three losses at once?** They are selectable variants. Compare
    them under matched conditions rather than stacking arbitrary objectives.
15. **How do you combine the heads?** Average their supervised losses. Task weights
    could be tuned if task difficulty or gradient scales cause imbalance.
16. **Why macro-F1?** It treats each class equally, exposing weak rare-class recall and
    precision that accuracy on a dominant class can hide.
17. **What does 82% mean?** Specify the split, per-head results, averaging rule, sample
    count, and exact match. Our initial validation mean is 57.73%, and test mean is
    58.16%; the test exact match is 10.20%. The proposed 82% remains unsupported.
18. **How do you calculate a 15% improvement?** State the baseline and whether the
    improvement is relative or in percentage points. Use multiple seeds when practical.
19. **What is an ablation?** Change one component at a time, such as adding augmentation
    with the same CE loss, then changing loss with augmentation held fixed.
20. **Can augmentation harm colour classification?** Yes. Strong hue transformations
    can alter the target colour; use restrained changes and validate per-task effects.
21. **Why might season be hard to predict?** The annotation can reflect catalog context
    rather than visible evidence. Some attributes have unavoidable ambiguity.
22. **What is the difference between validation and test?** Validation supports choices;
    test estimates the final selected model's performance on held-out data.
23. **How would you improve robustness testing?** Evaluate real held-out domains,
    lighting, backgrounds, resolution, and product families. Synthetic transformations
    alone do not establish robustness across real datasets.
24. **How does serving avoid repeated loading?** API startup loads one predictor;
    requests reuse it and its label map under controlled inference access.
25. **How did you measure latency?** Report hardware, image size, batch size, warmup,
    number of requests, p50/p95, and whether HTTP/upload time is included. The initial
    real checkpoint measured 27 ms median and 163 ms p95 over 30 warm local CPU HTTP
    requests with 96-pixel model inputs. That does not cover cold starts or concurrency.
26. **Can confidence be read as certainty?** Softmax confidence need not be calibrated.
    Use a held-out calibration set before treating it as a reliable probability.
27. **What happens to bad uploads?** The API rejects malformed, oversized, or unsuitable
    images with a clear HTTP response rather than calling the model on invalid data.
28. **What would you monitor after deployment?** Errors, p95 latency, data drift,
    prediction distributions, and labelled production samples for per-task quality.
29. **What are the current limitations?** CPU-only cloud resources, small source images,
    possible near-duplicate leakage, label ambiguity, and any outstanding real-data runs.
30. **What did you personally learn?** Walk through one audit issue, one modelling
    tradeoff, and one measured experiment. Use the build log and actual outputs.

## Equations to understand

For a head with logits `z`, `p = softmax(z)` and true class `y`:

- Cross-entropy: `L = -log(p[y])`.
- Weighted CE: `L = -w[y] * log(p[y])`, with weights computed from training support.
- Focal loss: `L = -w[y] * (1 - p[y])^gamma * log(p[y])`.
- Smoothed targets: `q = (1 - epsilon) * one_hot(y) + epsilon / C`.
- Multi-task objective: `L_total = mean(L_gender, L_article, L_colour, L_season)`.

The exact reduction and weighting in the implementation matter; read `losses.py`.

## Before adding résumé numbers

Keep the audit, split seed, checkpoint configuration, evaluation JSON, ablation
comparison, and latency report. Cite a real held-out metric and hardware. If only the
demo has run, use an honest implementation statement:

“Implemented and tested a reproducible four-task ResNet50 pipeline with data auditing,
three custom loss alternatives, augmentation, and FastAPI/Streamlit inference.”
