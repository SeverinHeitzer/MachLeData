"""Command-line entry point for local model training."""

from __future__ import annotations

import argparse
import json

from machledata.orchestration import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    config_value,
    load_yaml_config,
    prepare_dataset,
    train_model,
)


def main() -> None:
    """Prepare a dataset descriptor and create placeholder training outputs."""
    data_config = load_yaml_config(DEFAULT_DATA_CONFIG)
    model_config = load_yaml_config(DEFAULT_MODEL_CONFIG)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=config_value(data_config, "dataset", "local-demo"))
    parser.add_argument("--model-name", default=config_value(model_config, "model_name", "yolov8n"))
    parser.add_argument("--epochs", type=int, default=int(config_value(model_config, "epochs", 10)))
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-label", default="cli-train")
    args = parser.parse_args()

    prepared_dataset = prepare_dataset(
        dataset_id=args.dataset_id,
        artifact_root=args.artifact_root,
        samples_dir=config_value(data_config, "samples_dir", "data/samples"),
        project_id=config_value(data_config, "project_id"),
        bigquery_dataset=config_value(data_config, "dataset"),
        run_label=args.run_label,
    )
    run = train_model(
        prepared_dataset=prepared_dataset,
        model_name=args.model_name,
        epochs=args.epochs,
        artifact_root=args.artifact_root,
        run_label=args.run_label,
    )
    print(json.dumps(run, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
