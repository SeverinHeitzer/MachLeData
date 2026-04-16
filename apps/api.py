"""FastAPI application for serving object detection predictions.

The API should expose lightweight inference endpoints backed by
`machledata.infer` and avoid loading cloud credentials at import time.
"""

from fastapi import FastAPI

from machledata.infer import Detection

app = FastAPI(title="MachLeData Object Detection API")


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health for local checks and container probes."""
    return {"status": "ok"}


@app.post("/predict")
def predict() -> list[Detection]:
    """Return object detections for an uploaded image once inference is wired in."""
    return []

