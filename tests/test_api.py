"""Tests for the FastAPI application endpoints."""

from fastapi.testclient import TestClient

from apps.api import app


def test_health_endpoint() -> None:
    """Health endpoint returns a simple status payload for probes."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_dashboard_contract() -> None:
    """Prediction endpoint returns the shared response object."""
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.jpg", b"not-an-image", "image/jpeg")},
            params={"return_annotated": "true", "confidence_threshold": "0.5"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "annotated_image_base64": None,
        "detections": [],
    }
