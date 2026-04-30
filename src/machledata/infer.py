"""Inference helpers and schemas for YOLO object detection."""

import threading
from pathlib import Path

import torch
from pydantic import BaseModel, Field
from ultralytics import YOLO

from machledata.model import ModelConfig, build_model_config, load_model, load_saved_model


class Detection(BaseModel):
    """Single object detection result returned by the inference layer."""

    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]


class PredictionResponse(BaseModel):
    """Response returned by the API and consumed by the dashboard."""

    detections: list[Detection]


_model_cache: dict = {}
_model_cache_lock = threading.Lock()


def predict_image(
    image_path: str | Path,
    model_path: str | Path | None = None,
    config: ModelConfig | None = None,
) -> list[Detection]:
    """Run object detection for one image.

    Args:
        image_path: Local path to an image file.
        model_path: Path to a saved YOLO model. If None, uses default pretrained model.
        config: ModelConfig for inference settings. Uses defaults if None.

    Returns:
        Detection results with labels, confidence scores, and bounding boxes.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return []

    if config is None:
        config = build_model_config()

    model_key = str(model_path or config.model_name)
    with _model_cache_lock:
        if model_key not in _model_cache:
            if model_path and Path(model_path).exists():
                _model_cache[model_key] = load_saved_model(model_path)
            else:
                _model_cache[model_key] = load_model(config)
        model = _model_cache[model_key]

    # Run inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = model.predict(
        source=str(image_path),
        imgsz=config.image_size,
        conf=config.confidence_threshold,
        device=device,
        verbose=False,
    )

    # Parse results into Detection objects
    detections = []
    if results:
        result = results[0]
        boxes = result.boxes

        for box in boxes:
            # Extract coordinates and confidence
            xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy, 'tolist') else box.xyxy[0]
            conf = float(box.conf[0]) if hasattr(box.conf, '__getitem__') else float(box.conf)
            cls_id = int(box.cls[0]) if hasattr(box.cls, '__getitem__') else int(box.cls)

            # Get class name
            label = result.names.get(cls_id, f"class_{cls_id}") if result.names else f"class_{cls_id}"

            detections.append(
                Detection(
                    class_name=label,
                    confidence=conf,
                    bbox=tuple(xyxy),
                )
            )

    return detections


def predict_batch(
    image_paths: list[str | Path],
    model_path: str | Path | None = None,
    config: ModelConfig | None = None,
) -> dict[str, list[Detection]]:
    """Run object detection on multiple images.

    Args:
        image_paths: List of paths to image files.
        model_path: Path to a saved YOLO model.
        config: ModelConfig for inference settings.

    Returns:
        Dictionary mapping image paths to detection lists.
    """
    results = {}
    for image_path in image_paths:
        results[str(image_path)] = predict_image(image_path, model_path, config)
    return results


def clear_model_cache() -> None:
    """Clear the global model cache to free memory."""
    with _model_cache_lock:
        _model_cache.clear()
