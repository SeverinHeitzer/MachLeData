"""Tests for the FastAPI application endpoints."""

from fastapi.testclient import TestClient

from apps.api import app


def test_health_endpoint() -> None:
    """Health endpoint returns a simple status payload for probes."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

