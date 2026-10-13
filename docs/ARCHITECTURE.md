# Understand the architecture

```mermaid
flowchart LR
    A[Product image] --> B[RGB resize and normalization]
    B --> C[Shared ResNet50]
    C --> D[2048 visual features]
    D --> E[Gender: 5 logits]
    D --> F[Article type: 142 logits]
    D --> G[Colour: 47 logits]
    D --> H[Season: 4 logits]
    E --> I[Per-head softmax and label mapping]
    F --> I
    G --> I
    H --> I
    I --> J[FastAPI response]
    J --> K[Streamlit predictions]
```

The class counts above come from the actual prepared public dataset. They are not
hard-coded into the model: `labels.json` determines each head's dimensions and index
ordering. Synthetic or custom datasets can have different class counts.

Training consumes logits directly, because cross-entropy includes log-softmax.
Inference applies softmax to expose the highest-probability class and confidence.

```mermaid
flowchart LR
    A[Metadata and images] --> B[Audit and duplicate grouping]
    B --> C[Training split]
    B --> D[Validation split]
    B --> E[Held-out test split]
    C --> F[Vocabulary and class weights]
    C --> G[Training augmentation]
    F --> H[Train model]
    G --> H
    D --> I[Validation macro-F1]
    H --> I
    I --> J[Select checkpoint]
    J --> K[Final evaluation]
    E --> K
```

The initial CPU workflow freezes ResNet50 and updates only the heads. Batch
normalization in the frozen backbone remains in evaluation mode. A later fine-tuning
run can initialize from that checkpoint, unfreeze the backbone, and start a fresh
optimizer at a smaller learning rate. Initialization does not restore optimizer state.

Relevant code: `data.py` prepares inputs, `model.py` defines the shared network,
`losses.py` defines objectives, `train.py` selects checkpoints, `evaluate.py` computes
held-out metrics, and `inference.py` restores the model for both benchmark and API use.
