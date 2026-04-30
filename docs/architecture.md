# Architecture Notes

## End-to-End Shape

MachLeData uses a thin-interface pattern:

- `src/machledata/` owns reusable data, training, evaluation, inference, and orchestration code.
- `machledata.pipeline_steps` adapts that package code to containerized Kubeflow steps.
- `workflows/kubeflow_pipeline.py` describes the pipeline graph without embedding ML logic.
- `apps/api.py` and `apps/dashboard.py` are downstream consumers of publish-ready artifacts and the shared prediction contract.

The code is intentionally prepared so the orchestration layer should not need to change when the real model arrives. Model-specific work belongs behind the package seams in `machledata.data`, `machledata.train`, `machledata.infer`, and `machledata.metrics`.

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

Pipeline step adapters are exposed through `python -m machledata.pipeline_steps`. They accept simple command-line values and file paths, then read or write JSON payloads. This makes the same code usable from local smoke tests, Kubeflow containers, and Vertex AI Pipelines.

Simple pipeline diagram:

```text
prepare-data -> train-model -> evaluate-model -> publish-artifact-metadata
```

## Vertex AI Handoff

The first Kubeflow version intentionally stops at validated, publish-ready outputs. That handoff boundary is the `artifact_manifest.json` produced by the publish stage.

For Vertex AI Pipelines, artifacts should live under a `gs://` pipeline root supplied at submission time. The local placeholder `artifact_root` remains useful for CLI smoke tests and can later be replaced by a GCS-backed path when the real model and dataset are wired in.

The recommended flow is:

1. Build and push the project image to a registry Vertex AI can access.
2. Compile the pipeline with that exact image URI.
3. Submit the compiled YAML with project, region, pipeline root, and optional service account.

```bash
python scripts/compile_pipeline.py \
  --image-uri "$VERTEX_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/machledata/machledata:latest"

python scripts/submit_vertex_pipeline.py \
  --project-id "$GOOGLE_CLOUD_PROJECT" \
  --region "$VERTEX_REGION" \
  --pipeline-root "$VERTEX_PIPELINE_ROOT" \
  --template-path artifacts/pipelines/machledata_pipeline.yaml
```

Future GCP integration points:

- BigQuery metadata and image references in the data preparation step
- model artifacts stored in GCS or Vertex Model Registry
- metadata read by a FastAPI inference service
- reports surfaced through the Streamlit dashboard
- optional deployment orchestration through Vertex AI, Cloud Run, or CI/CD

Website hosting and endpoint rollout are outside the initial pipeline scope.

## Prediction Contract

The API and dashboard use one shared prediction shape:

```json
{
  "detections": [
    {
      "class_name": "object",
      "confidence": 0.9,
      "bbox": [0.0, 0.0, 10.0, 10.0]
    }
  ],
  "annotated_image_base64": null
}
```

`machledata.infer.predict_image` is the intended model integration point. It currently returns an empty list and should later load the selected model artifact, run inference, and return `Detection` objects with the same field names.

## Local Runtime

`docker/docker-compose.yml` runs the API service for local app work. Kubeflow pipeline compilation is performed from the Python environment with:

```bash
python scripts/compile_pipeline.py --image-uri machledata:local
```

The compiled YAML can be uploaded to a Kubeflow-compatible backend or submitted to Vertex AI with `scripts/submit_vertex_pipeline.py`.

Local CLI checks use `/tmp` or `artifacts/` for generated outputs. Those generated outputs are intentionally outside the tracked project state.
