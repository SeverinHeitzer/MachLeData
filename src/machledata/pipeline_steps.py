"""CLI adapters used by Kubeflow container components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from machledata.orchestration import (
    DEFAULT_APP_CONFIG,
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
    """Dispatch one pipeline step and write its JSON output."""
    import sys
    print("=== MACHLEDATA STEP START ===", flush=True)
    print(f"argv: {sys.argv}", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--dataset-id", default=None)
    prepare.add_argument("--samples-dir", default=None)
    prepare.add_argument("--project-id", default=None)
    prepare.add_argument("--bigquery-dataset", default=None)
    prepare.add_argument("--images-table", default=None)
    prepare.add_argument("--labels-table", default=None)
    prepare.add_argument("--split", default=None)
    prepare.add_argument("--max-rows", type=int, default=None)
    prepare.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    prepare.add_argument("--run-label", default=None)
    prepare.add_argument("--output-path", required=True)

    train = subparsers.add_parser("train-model")
    train.add_argument("--prepared-dataset-path", required=True)
    train.add_argument("--model-name", default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    train.add_argument("--run-label", default=None)
    train.add_argument("--model-output-path", required=True)
    train.add_argument("--metadata-output-path", required=True)

    evaluate = subparsers.add_parser("evaluate-model")
    evaluate.add_argument("--prepared-dataset-path", required=True)
    evaluate.add_argument("--training-run-path", required=True)
    evaluate.add_argument("--model-artifact-path", required=True)
    evaluate.add_argument("--min-detections-per-image", type=float, default=0.1)
    evaluate.add_argument("--evaluation-output-path", required=True)
    evaluate.add_argument("--metrics-output-path", required=True)

    publish = subparsers.add_parser("publish-artifact-metadata")
    publish.add_argument("--prepared-dataset-path", required=True)
    publish.add_argument("--training-run-path", required=True)
    publish.add_argument("--evaluation-summary-path", required=True)
    publish.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    publish.add_argument("--manifest-output-path", required=True)

    args = parser.parse_args()
    if args.command == "prepare-data":
        _write_json(Path(args.output_path), _run_prepare(args))
    elif args.command == "train-model":
        _write_json(Path(args.metadata_output_path), _run_train(args))
    elif args.command == "evaluate-model":
        _write_json(Path(args.evaluation_output_path), _run_evaluate(args))
    elif args.command == "publish-artifact-metadata":
        _write_json(Path(args.manifest_output_path), _run_publish(args))


def _run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    print("=== _run_prepare entered ===", flush=True)
    data_config = load_yaml_config(DEFAULT_DATA_CONFIG)
    dataset_id = _clean(args.dataset_id) or config_value(
        data_config,
        "dataset",
        "local-demo",
    )
    return prepare_dataset(
        dataset_id=dataset_id,
        artifact_root=args.artifact_root,
        samples_dir=_clean(args.samples_dir)
        or config_value(data_config, "samples_dir", "data/samples"),
        project_id=_clean(args.project_id) or config_value(data_config, "project_id"),
        bigquery_dataset=_clean(args.bigquery_dataset)
        or config_value(data_config, "dataset"),
        images_table=_clean(args.images_table)
        or config_value(data_config, "images_table"),
        labels_table=_clean(args.labels_table)
        or config_value(data_config, "labels_table"),
        split=_clean(args.split) or config_value(data_config, "split"),
        max_rows=args.max_rows if args.max_rows and args.max_rows > 0 else None,
        run_label=_clean(args.run_label),
        descriptor_output_path=args.output_path,
    )


def _run_train(args: argparse.Namespace) -> dict[str, Any]:
    print("=== _run_train entered ===", flush=True)
    model_config = load_yaml_config(DEFAULT_MODEL_CONFIG)
    prepared = _read_json(Path(args.prepared_dataset_path))
    print(f"prepared keys: {list(prepared.keys())}", flush=True)
    print(f"yolo_dataset_yaml: {prepared.get('yolo_dataset_yaml')}", flush=True)
    print(f"annotation_count: {prepared.get('annotation_count')}", flush=True)
    return train_model(
        prepared_dataset=prepared,
        model_name=_clean(args.model_name)
        or config_value(model_config, "model_name", "yolov8n"),
        epochs=args.epochs or int(config_value(model_config, "epochs", 10)),
        artifact_root=args.artifact_root,
        run_label=_clean(args.run_label),
        model_artifact_path=args.model_output_path,
        training_metadata_path=args.metadata_output_path,
    )

def _run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    return evaluate_model(
        prepared_dataset=_read_json(Path(args.prepared_dataset_path)),
        training_run=_read_json(Path(args.training_run_path)),
        min_detections_per_image=args.min_detections_per_image,
        model_artifact_path=args.model_artifact_path,
        evaluation_output_path=args.evaluation_output_path,
        metrics_output_path=args.metrics_output_path,
    )


def _run_publish(args: argparse.Namespace) -> dict[str, Any]:
    return publish_artifact_manifest(
        prepared_dataset=_read_json(Path(args.prepared_dataset_path)),
        training_run=_read_json(Path(args.training_run_path)),
        evaluation_summary=_read_json(Path(args.evaluation_summary_path)),
        artifact_root=args.artifact_root,
        app_config_path=DEFAULT_APP_CONFIG,
        manifest_output_path=args.manifest_output_path,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _clean(value: str | None) -> str | None:
    return value if value and not value.startswith("${") else None


if __name__ == "__main__":
    main()
