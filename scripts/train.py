"""Command-line entry point for local model training.

This script should read `configs/model.yaml`, prepare training metadata, and
delegate the actual training work to `machledata.train`.
"""

from machledata.train import create_training_run


def main() -> None:
    """Create a placeholder training run until full training is implemented."""
    run = create_training_run("yolov8n", epochs=10, artifact_dir="artifacts/models")
    print(run)


if __name__ == "__main__":
    main()

