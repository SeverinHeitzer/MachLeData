"""Tests for the FastAPI application endpoints."""

from fastapi.testclient import TestClient

from apps.api import app


def test_health_endpoint() -> None:
    """Health endpoint returns a simple status payload for probes."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_with_invalid_image() -> None:
    """Prediction endpoint returns 400 for invalid image data."""
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("sample.jpg", b"not-an-image", "image/jpeg")},
            params={"confidence_threshold": "0.5"},
        )

    assert response.status_code == 400
    assert "detail" in response.json()

