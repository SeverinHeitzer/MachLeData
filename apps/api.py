"""FastAPI application for serving object detection predictions."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from machledata.config import load_yaml_config
from machledata.infer import PredictionResponse, predict_image
from machledata.model import build_model_config

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_app_config() -> dict:
    """Load application serving configuration."""
    return load_yaml_config("configs/app.yaml")

_app_config = load_app_config()
app = FastAPI(title=_app_config.get("api_title", "MachLeData Object Detection API"))

# Default model configuration (now loads from configs/model.yaml and configs/app.yaml)
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
    confidence_threshold: float | None = None,
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
                img = Image.open(tmppath)
                img.verify()  # Check if it's a valid image file
                # Re-open because verify() closes the file
                Image.open(tmppath)
            except Exception as e:
                logger.warning(f"Invalid image uploaded: {e}")
                raise HTTPException(status_code=400, detail="Invalid image file")

            # Run inference
            config = _default_config
            if confidence_threshold is not None and confidence_threshold != _default_config.confidence_threshold:
                config = build_model_config(
                    model_name=_default_config.model_name,
                    image_size=_default_config.image_size,
                    confidence_threshold=confidence_threshold,
                )

            logger.info(f"Running inference on {file.filename} with conf={config.confidence_threshold}")
            detections = predict_image(tmppath, config=config)
            return PredictionResponse(detections=detections)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
