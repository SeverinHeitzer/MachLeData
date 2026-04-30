"""Inference helpers and schemas for YOLO object detection."""

from pathlib import Path

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Single object detection result returned by the inference layer."""

    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]


class PredictionResponse(BaseModel):
    """Response returned by the API and consumed by the dashboard."""

    detections: list[Detection]
    annotated_image_base64: str | None = None


def predict_image(image_path: str | Path) -> list[Detection]:
    """Run object detection for one image.

    Args:
        image_path: Local path to an image file.

    Returns:
        Detection results. The skeleton returns an empty list until a model is wired in.
    """
    _ = Path(image_path)
    return []
