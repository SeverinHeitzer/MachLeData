"""Model loading and configuration helpers for YOLO object detection.

This module wraps the Ultralytics YOLO library with Hugging Face integration,
providing a stable model interface for the rest of the project.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics import YOLO


@dataclass(frozen=True)
class ModelConfig:
    """Configuration values needed to select and run an object detection model."""

    model_name: str
    image_size: int
    confidence_threshold: float = 0.25


def build_model_config(
    model_name: str = "yolov8n",
    image_size: int = 640,
    confidence_threshold: float = 0.25,
) -> ModelConfig:
    """Create a typed model configuration for scripts and applications."""
    return ModelConfig(
        model_name=model_name,
        image_size=image_size,
        confidence_threshold=confidence_threshold,
    )


def load_model(config: ModelConfig, device: str | None = None) -> YOLO:
    """Load a YOLO model from Ultralytics/Hugging Face.

    Args:
        config: ModelConfig with model_name and configuration.
        device: Device to load model on ('cpu', 'cuda', 'mps', etc.).
                Auto-detects if None.

    Returns:
        Loaded YOLO model ready for inference.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model from Ultralytics (which supports HF model hub)
    model = YOLO(config.model_name)
    model = model.to(device)
    return model


def save_model(model: YOLO, output_path: str | Path) -> None:
    """Save a trained YOLO model to disk.

    Args:
        model: Trained YOLO model.
        output_path: Path where the model will be saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))


def load_saved_model(model_path: str | Path, device: str | None = None) -> YOLO:
    """Load a previously saved YOLO model.

    Args:
        model_path: Path to saved model file.
        device: Device to load model on.

    Returns:
        Loaded YOLO model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(str(model_path))
    model = model.to(device)
    return model


