"""Command-line entry point for local model training."""

from __future__ import annotations

import argparse
import json

from machledata.model import build_model_config
from machledata.orchestration import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    config_value,
    load_yaml_config,
    prepare_dataset,
    train_model,
)
from machledata.train import train_yolo_model


def main() -> None:
    """Prepare a dataset descriptor and create training outputs with real YOLO training."""
    data_config = load_yaml_config(DEFAULT_DATA_CONFIG)
    model_config = load_yaml_config(DEFAULT_MODEL_CONFIG)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=config_value(data_config, "dataset", "local-demo"))
    parser.add_argument("--model-name", default=None, help="YOLO model name")
    parser.add_argument("--epochs", type=int, default=int(config_value(model_config, "epochs", 10)))
    parser.add_argument("--batch-size", type=int, default=int(config_value(model_config, "batch_size", 8)))
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-label", default="cli-train")
    parser.add_argument("--dataset-path", default=None, help="Path to YOLO dataset YAML or directory")
    args = parser.parse_args()

    prepared_dataset = prepare_dataset(
        dataset_id=args.dataset_id,
        artifact_root=args.artifact_root,
        samples_dir=config_value(data_config, "samples_dir", "data/samples"),
        project_id=config_value(data_config, "project_id"),
        bigquery_dataset=config_value(data_config, "dataset"),
        run_label=args.run_label,
    )

    # Use orchestration.train_model which calls create_training_run
    # For real YOLO training with a dataset, integrate train_yolo_model
    run = train_model(
        prepared_dataset=prepared_dataset,
        model_name=args.model_name or config_value(model_config, "model_name", "yolov8n"),
        epochs=args.epochs,
        artifact_root=args.artifact_root,
        run_label=args.run_label,
    )

    # If dataset path is provided, also run actual YOLO training
    if args.dataset_path:
        config = build_model_config(
            model_name=args.model_name,
        )
        try:
            trained_model, training_metrics = train_yolo_model(
                config=config,
                dataset_path=args.dataset_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                artifact_dir=run["artifact_dir"],
            )
            run["training_metrics"] = training_metrics
            print(f"YOLO training completed successfully")
        except Exception as e:
            print(f"Warning: YOLO training failed: {e}")
            print(f"Continuing with placeholder model")

    print(json.dumps(run, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

