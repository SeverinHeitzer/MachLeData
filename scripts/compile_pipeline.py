"""Compile the MachLeData Kubeflow pipeline to a YAML package."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from kfp import compiler

from machledata.orchestration import config_value, load_yaml_config


PIPELINE_CONFIG = "configs/pipeline.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Compile the Kubeflow pipeline for upload to Vertex AI Pipelines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-uri", default=None)
    parser.add_argument("--package-path", default=None)
    args = parser.parse_args()

    config = load_yaml_config(PIPELINE_CONFIG)
    image_uri = args.image_uri or config_value(config, "image_uri", "machledata:local")
    package_path = Path(
        args.package_path
        or config_value(config, "pipeline_package_path", "artifacts/pipelines/machledata_pipeline.yaml")
    )
    package_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["MACHLEDATA_PIPELINE_IMAGE"] = image_uri
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from workflows.kubeflow_pipeline import machledata_pipeline

    compiler.Compiler().compile(
        pipeline_func=machledata_pipeline,
        package_path=str(package_path),
    )
    print(f"Compiled Kubeflow pipeline to {package_path}")


if __name__ == "__main__":
    main()
