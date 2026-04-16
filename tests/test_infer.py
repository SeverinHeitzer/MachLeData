"""Tests for the inference interface used by scripts and applications."""

from machledata.infer import predict_image


def test_predict_image_returns_list() -> None:
    """The skeleton predictor returns a list before the model is wired in."""
    assert predict_image("data/samples/example.jpg") == []

