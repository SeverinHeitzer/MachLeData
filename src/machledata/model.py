"""Model loading and configuration helpers for YOLO object detection.

This module will later wrap the chosen PyTorch, Hugging Face, or YOLO library
so the rest of the project can use a stable model interface.
"""

from dataclasses import dataclass


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

