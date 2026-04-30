"""FastAPI application for serving object detection predictions."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

from machledata.infer import PredictionResponse, predict_image

app = FastAPI(title="MachLeData Object Detection API")


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health for local checks and container probes."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile | None = File(default=None),
    return_annotated: bool = False,
    confidence_threshold: float = 0.25,
) -> PredictionResponse:
    """Return object detections for an uploaded image once inference is wired in."""
    _ = return_annotated, confidence_threshold
    if file is None:
        return PredictionResponse(detections=[])

    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        detections = predict_image(tmp.name)
    return PredictionResponse(detections=detections)
