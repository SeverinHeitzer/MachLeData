"""Tests for the inference interface used by scripts and applications."""

from machledata.infer import Detection, PredictionResponse, predict_image


def test_predict_image_returns_list() -> None:
    """The skeleton predictor returns a list before the model is wired in."""
    assert predict_image("data/samples/example.jpg") == []


def test_prediction_schema_matches_api_and_dashboard_contract() -> None:
    """Prediction responses expose the fields consumed by the dashboard."""
    response = PredictionResponse(
        detections=[
            Detection(
                class_name="example",
                confidence=0.8,
                bbox=(1.0, 2.0, 3.0, 4.0),
            )
        ],
    )

    assert response.model_dump() == {
        "annotated_image_base64": None,
        "detections": [
            {
                "bbox": (1.0, 2.0, 3.0, 4.0),
                "class_name": "example",
                "confidence": 0.8,
            }
        ],
    }
