# Architecture Notes

## End-to-End Shape

MachLeData uses a thin-interface pattern:

- `src/machledata/` owns reusable data, training, evaluation, inference, and orchestration code.
- `machledata.pipeline_steps` adapts that package code to containerized Kubeflow steps.
- `workflows/kubeflow_pipeline.py` describes the pipeline graph without embedding ML logic.
- `apps/api.py` and `apps/dashboard.py` are downstream consumers of publish-ready artifacts and the shared prediction contract.

## Kubeflow Pipeline

The canonical pipeline is `machledata-ml-pipeline`. It is defined with Kubeflow Pipelines v2 and is intended to run locally as compiled YAML or remotely through Vertex AI Pipelines.

Pipeline stages:

1. `prepare-data`
2. `train-model`
3. `evaluate-model`
4. `publish-artifact-metadata`

Stage contract:

- `prepare-data` writes a dataset descriptor with source information, sample inventory, and a config snapshot.
- `train-model` writes training metadata plus a model artifact placeholder path.
- `evaluate-model` writes an evaluation summary with metrics and a pass/fail gate.
- `publish-artifact-metadata` writes a serving-facing manifest only when evaluation passes.

Simple pipeline diagram:

```text
prepare-data -> train-model -> evaluate-model -> publish-artifact-metadata
```

## Vertex AI Handoff

The first Kubeflow version intentionally stops at validated, publish-ready outputs. That handoff boundary is the `artifact_manifest.json` produced by the publish stage.

For Vertex AI Pipelines, artifacts should live under a `gs://` pipeline root supplied at submission time. The local placeholder `artifact_root` remains useful for CLI smoke tests and can later be replaced by a GCS-backed path when the real model and dataset are wired in.

Future GCP integration points:

- BigQuery metadata and image references in the data preparation step
- model artifacts stored in GCS or Vertex Model Registry
- metadata read by a FastAPI inference service
- reports surfaced through the Streamlit dashboard
- optional deployment orchestration through Vertex AI, Cloud Run, or CI/CD

Website hosting and endpoint rollout are outside the initial pipeline scope.

## Local Runtime

`docker/docker-compose.yml` runs the API service for local app work. Kubeflow pipeline compilation is performed from the Python environment with:

```bash
python scripts/compile_pipeline.py --image-uri machledata:local
```

The compiled YAML can be uploaded to a Kubeflow-compatible backend or submitted to Vertex AI with `scripts/submit_vertex_pipeline.py`.
