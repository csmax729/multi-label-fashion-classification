"""Local checkpoint loading and the shared API/Streamlit prediction interface."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image

from fashion_classifier.data import TASKS, make_transforms
from fashion_classifier.model import ARCHITECTURE, FashionMultiTaskModel, input_memory_format


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {device}")
    return device


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[FashionMultiTaskModel, dict]:
    """Load only an application-generated, trusted local artifact.

    ``weights_only=True`` restricts deserialization to tensors and simple types.
    It is still inappropriate to accept arbitrary uploaded checkpoints.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Train a model first.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"schema_version", "architecture", "state_dict", "labels", "config", "provenance"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Checkpoint is missing required application metadata")
    if checkpoint["schema_version"] != 1 or checkpoint["architecture"] != ARCHITECTURE:
        raise ValueError("Unsupported checkpoint format or model architecture")
    labels = checkpoint["labels"]
    if not isinstance(labels, dict) or set(labels) != set(TASKS):
        raise ValueError("Checkpoint must contain all four task vocabularies")
    for task, vocabulary in labels.items():
        if (
            not isinstance(vocabulary, list)
            or not vocabulary
            or not all(isinstance(value, str) for value in vocabulary)
        ):
            raise ValueError(f"Invalid checkpoint vocabulary for {task}")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError(f"Duplicate checkpoint labels for {task}")
    image_size = checkpoint["config"].get("image_size")
    if not isinstance(image_size, int) or image_size < 32 or image_size > 2048:
        raise ValueError("Checkpoint image_size must be an integer between 32 and 2048")
    model = FashionMultiTaskModel({task: len(labels[task]) for task in TASKS}, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    memory_format = (
        torch.channels_last
        if checkpoint["config"].get("channels_last", False)
        else torch.contiguous_format
    )
    model.to(device, memory_format=memory_format)
    model.eval()
    return model, checkpoint


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        self.device = resolve_device(device)
        # Cloud hosts can expose hundreds of cores; batch-one inference suffers
        # when the default thread count oversubscribes such a host.
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.model, metadata = load_checkpoint(checkpoint, self.device)
        self.memory_format = input_memory_format(self.model)
        self.labels = metadata["labels"]
        self.transform = make_transforms(metadata["config"]["image_size"], augment=False)
        provenance = metadata["provenance"]
        self.model_info = {
            "demo": bool(provenance.get("demo", False)),
            "architecture": ARCHITECTURE,
            "dataset_source": provenance.get("dataset_source", "unverified"),
            "dataset_kind": provenance.get("dataset_kind", "unverified"),
            "backbone_weight_source": provenance.get("backbone_weight_source", "unknown"),
            "checkpoint_path": str(Path(checkpoint).resolve()),
            "channels_last": self.memory_format == torch.channels_last,
        }

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        start = time.perf_counter()
        tensor = (
            self.transform(image.convert("RGB"))
            .unsqueeze(0)
            .to(self.device, memory_format=self.memory_format)
        )
        outputs = self.model(tensor)
        predictions = {}
        for task in TASKS:
            probabilities = outputs[task].softmax(dim=-1)[0]
            confidence, index = probabilities.max(dim=0)
            predictions[task] = {
                "label": self.labels[task][int(index)],
                "confidence": float(confidence),
            }
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "predictions": predictions,
            "latency_ms": (time.perf_counter() - start) * 1000.0,
            "model_info": dict(self.model_info),
        }
