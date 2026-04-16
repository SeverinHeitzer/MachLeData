"""Metric utilities for evaluating object detection performance.

This module will collect model quality metrics such as precision, recall, and
mean Average Precision, plus operational metrics used by the dashboard.
"""


def summarize_detections(total_images: int, total_detections: int) -> dict[str, float]:
    """Compute simple detection summary metrics for demos and smoke tests."""
    average = total_detections / total_images if total_images else 0.0
    return {
        "total_images": float(total_images),
        "total_detections": float(total_detections),
        "detections_per_image": average,
    }

