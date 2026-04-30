"""Submit a compiled Kubeflow pipeline to Vertex AI Pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

from machledata.orchestration import config_value, load_yaml_config


PIPELINE_CONFIG = "configs/pipeline.yaml"


def main() -> None:
    """Create and submit a Vertex AI PipelineJob."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--pipeline-root", default=None)
    parser.add_argument("--template-path", default=None)
    parser.add_argument("--service-account", default=None)
    parser.add_argument("--dataset-id", default="local-demo")
    parser.add_argument("--samples-dir", default="data/samples")
    parser.add_argument("--bigquery-dataset", default="")
    parser.add_argument("--images-table", default="images")
    parser.add_argument("--labels-table", default="labels")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--model-name", default="yolov8n")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--run-label", default="manual")
    parser.add_argument("--min-detections-per-image", type=float, default=0.1)
    args = parser.parse_args()

    config = load_yaml_config(PIPELINE_CONFIG)
    project_id = args.project_id or config_value(config, "project_id")
    region = args.region or config_value(config, "region")
    pipeline_root = args.pipeline_root or config_value(config, "pipeline_root")
    template_path = args.template_path or config_value(config, "pipeline_package_path")
    service_account = args.service_account or config_value(config, "service_account")
    artifact_root = args.artifact_root or config_value(config, "artifact_root", "/tmp/machledata-artifacts")

    missing = [
        name
        for name, value in {
            "project_id": project_id,
            "region": region,
            "pipeline_root": pipeline_root,
            "template_path": template_path,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required Vertex settings: {', '.join(missing)}")
    if not Path(template_path).exists():
        raise SystemExit(f"Pipeline template does not exist: {template_path}")

    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)
    job = aiplatform.PipelineJob(
        display_name="machledata-ml-pipeline",
        template_path=template_path,
        pipeline_root=pipeline_root,
        parameter_values={
            "dataset_id": args.dataset_id,
            "samples_dir": args.samples_dir,
            "project_id": project_id,
            "bigquery_dataset": args.bigquery_dataset,
            "images_table": args.images_table,
            "labels_table": args.labels_table,
            "split": args.split,
            "max_rows": args.max_rows,
            "model_name": args.model_name,
            "epochs": args.epochs,
            "artifact_root": artifact_root,
            "run_label": args.run_label,
            "min_detections_per_image": args.min_detections_per_image,
        },
    )
    job.submit(service_account=service_account or None)
    print(f"Submitted Vertex AI PipelineJob: {job.display_name}")


if __name__ == "__main__":
    main()
