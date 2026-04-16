"""Inference helpers for YOLO object detection.

Applications and scripts should call this module instead of depending directly
on model-library-specific prediction APIs.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Detection:
    """Single object detection result returned by the inference layer."""

    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


def predict_image(image_path: str | Path) -> list[Detection]:
    """Run object detection for one image.

    Args:
        image_path: Local path to an image file.

    Returns:
        Detection results. The skeleton returns an empty list until a model is wired in.
    """
    _ = Path(image_path)
    return []

