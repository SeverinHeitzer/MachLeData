"""FastAPI application for serving object detection predictions."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from machledata.infer import PredictionResponse, predict_image
from machledata.model import build_model_config

app = FastAPI(title="MachLeData Object Detection API")

# Default model configuration
_default_config = build_model_config()


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health for local checks and container probes."""
    return {"status": "ok"}


@app.get("/model/config")
def get_model_config() -> dict:
    """Return current model configuration."""
    return {
        "model_name": _default_config.model_name,
        "image_size": _default_config.image_size,
        "confidence_threshold": _default_config.confidence_threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile | None = File(default=None),
    confidence_threshold: float = 0.25,
) -> PredictionResponse:
    """Return object detections for an uploaded image."""
    if file is None:
        return PredictionResponse(detections=[])

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir) / (file.filename or "image.jpg")
            contents = await file.read()
            tmppath.write_bytes(contents)

            # Validate image
            try:
                Image.open(tmppath)
            except Exception:
                return PredictionResponse(detections=[])

            # Run inference with custom threshold if provided
            config = _default_config
            if confidence_threshold != _default_config.confidence_threshold:
                config = build_model_config(
                    model_name=_default_config.model_name,
                    image_size=_default_config.image_size,
                    confidence_threshold=confidence_threshold,
                )

            detections = predict_image(tmppath, config=config)
            return PredictionResponse(detections=detections)

    except Exception:
        return PredictionResponse(detections=[])
