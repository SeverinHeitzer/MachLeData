"""Command-line entry point for model evaluation and manifest publishing."""

from __future__ import annotations

import argparse
import json

from machledata.data import load_sample_paths
from machledata.metrics import compute_detection_statistics, evaluate_on_images
from machledata.model import build_model_config
from machledata.orchestration import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    config_value,
    evaluate_model,
    load_yaml_config,
    prepare_dataset,
    publish_artifact_manifest,
    train_model,
)


def main() -> None:
    """Run the evaluate and publish stages with real YOLO evaluation."""
    data_config = load_yaml_config(DEFAULT_DATA_CONFIG)
    model_config = load_yaml_config(DEFAULT_MODEL_CONFIG)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=config_value(data_config, "dataset", "local-demo"))
    parser.add_argument("--model-name", default=config_value(model_config, "model_name", "yolov8n"))
    parser.add_argument("--epochs", type=int, default=int(config_value(model_config, "epochs", 10)))
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-label", default="cli-evaluate")
    parser.add_argument("--model-path", default=None, help="Path to saved model")
    parser.add_argument("--eval-images", default="data/samples", help="Path to images for evaluation")
    args = parser.parse_args()

    prepared_dataset = prepare_dataset(
        dataset_id=args.dataset_id,
        artifact_root=args.artifact_root,
        samples_dir=config_value(data_config, "samples_dir", "data/samples"),
        project_id=config_value(data_config, "project_id"),
        bigquery_dataset=config_value(data_config, "dataset"),
        run_label=args.run_label,
    )

    training_run = train_model(
        prepared_dataset=prepared_dataset,
        model_name=args.model_name,
        epochs=args.epochs,
        artifact_root=args.artifact_root,
        run_label=args.run_label,
    )

    # Run YOLO evaluation if images are available
    eval_metrics = {}
    eval_image_paths = load_sample_paths(args.eval_images)

    if eval_image_paths:
        try:
            config = build_model_config(
                model_name=args.model_name,
                image_size=int(config_value(model_config, "image_size", 640)),
            )
            eval_metrics = evaluate_on_images(
                eval_image_paths,
                model_path=args.model_path or training_run["model_artifact_path"],
            )
            print(f"Evaluation completed on {len(eval_image_paths)} images")
        except Exception as e:
            print(f"Warning: YOLO evaluation failed: {e}")

    # Use orchestration evaluate_model for standard metrics
    evaluation_summary = evaluate_model(
        prepared_dataset=prepared_dataset,
        training_run=training_run,
    )

    # Add YOLO evaluation metrics if available
    if eval_metrics:
        evaluation_summary["yolo_metrics"] = eval_metrics

    # Publish if evaluation passed
    try:
        manifest = publish_artifact_manifest(
            prepared_dataset=prepared_dataset,
            training_run=training_run,
            evaluation_summary=evaluation_summary,
            artifact_root=args.artifact_root,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    except ValueError as e:
        print(f"Error: {e}")
        print(json.dumps(evaluation_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

