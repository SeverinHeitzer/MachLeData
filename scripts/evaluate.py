"""Command-line entry point for model evaluation.

This script should load predictions and labels, compute object detection
metrics, and write evaluation summaries for the dashboard or reports.
"""

from machledata.metrics import summarize_detections


def main() -> None:
    """Print placeholder evaluation metrics for a smoke-testable CLI."""
    print(summarize_detections(total_images=0, total_detections=0))


if __name__ == "__main__":
    main()

