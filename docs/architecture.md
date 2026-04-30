# Architecture Notes

## End-to-End Shape

MachLeData uses a thin-interface pattern:

- `src/machledata/` owns reusable data, training, evaluation, inference, and orchestration code.
- `machledata.pipeline_steps` adapts that package code to containerized Kubeflow steps.
- `workflows/kubeflow_pipeline.py` describes the pipeline graph without embedding ML logic.
- `apps/api.py` and `apps/dashboard.py` are downstream consumers of publish-ready artifacts and the shared prediction contract.

The orchestration layer is designed to require no changes when the model or dataset evolves. Model-specific work belongs behind the package seams in `machledata.data`, `machledata.train`, `machledata.infer`, and `machledata.metrics`.

## Kubeflow Pipeline

The canonical pipeline is `machledata-ml-pipeline`. It is defined with Kubeflow Pipelines v2 and runs locally as a compiled YAML or remotely through Vertex AI Pipelines.

Pipeline stages:

| Stage | Input artifacts | Output artifacts |
|---|---|---|
| `prepare-data` | — | `system.Dataset` descriptor |
| `train-model` | `system.Dataset` | `system.Model` + `system.Artifact` training metadata |
| `evaluate-model` | `system.Dataset`, `system.Model`, `system.Artifact` | `system.Artifact` evaluation summary + `system.Metrics` |
| `publish-artifact-metadata` | all upstream artifacts | `system.Artifact` serving manifest |

Each stage runs the shared Docker image via `python -m machledata.pipeline_steps <command>`. The CLI adapter accepts simple command-line values and KFP-provided artifact paths, reads or writes JSON payloads at those paths, and returns. This keeps local CLI smoke tests, Kubeflow containers, and Vertex AI Pipelines aligned while preserving the typed artifact graph.

```
prepare-data → train-model → evaluate-model → publish-artifact-metadata
     ↓               ↓               ↓                    ↓
  Dataset       Model + Meta    Summary + Metrics       Manifest
```

## Vertex AI Handoff

The publish stage is the handoff boundary. It writes a manifest artifact only when evaluation passes. That manifest is the input for any downstream inference service or deployment step.

Recommended release flow:

1. Build and push the project image to a registry Vertex AI can access (Artifact Registry).
2. Compile the pipeline with that exact image URI.
3. Submit the compiled YAML with project, region, pipeline root, and optional service account.

```bash
python scripts/compile_pipeline.py \
  --image-uri "$VERTEX_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/machledata/machledata:latest"

python scripts/submit_vertex_pipeline.py \
  --project-id "$GOOGLE_CLOUD_PROJECT" \
  --region "$VERTEX_REGION" \
  --pipeline-root "$VERTEX_PIPELINE_ROOT" \
  --template-path artifacts/pipelines/machledata_pipeline.yaml \
  --bigquery-dataset "$BIGQUERY_DATASET" \
  --images-table images \
  --labels-table labels \
  --split train
```

Remaining GCP integration steps:

- BigQuery dataset populated with real image metadata and bounding-box annotations; image URIs pointing to accessible GCS paths
- YOLO `dataset.yaml` generated during `prepare-data` so `machledata.train.train_yolo_model` can be called from the `train-model` step (currently writes a placeholder artifact)
- Trained model artifacts stored in GCS or registered in Vertex Model Registry
- FastAPI inference service reading the published model artifact from GCS

## Prediction Contract

The API and dashboard share one detection schema:

```json
{
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.87,
      "bbox": [120.0, 45.0, 380.0, 510.0]
    }
  ]
}
```

`bbox` is `[x1, y1, x2, y2]` in pixel coordinates. `machledata.infer.Detection` is the authoritative type; both `apps/api.py` and `apps/dashboard.py` import it directly.

`machledata.infer.predict_image` is the model integration point. It loads a YOLOv8 model (from a saved artifact path or the default pretrained weights), runs inference, and returns a list of `Detection` objects. A thread-safe model cache avoids repeated disk reads across concurrent requests.

## Runtime Containers

The Docker image in `docker/Dockerfile` is built from `python:3.11-slim`, runs as a non-root user (`appuser`), and exposes port 8000. The default command starts the FastAPI service with uvicorn. The same image is used for:

- local API and dashboard development via `docker/docker-compose.yml`
- Kubeflow pipeline step containers (each stage runs `python -m machledata.pipeline_steps <command>`)

`docker/docker-compose.yml` runs the API service for local development. Kubeflow pipeline compilation is performed from the Python environment:

```bash
python scripts/compile_pipeline.py --image-uri machledata:local
```

The compiled YAML can be uploaded to a Kubeflow-compatible backend or submitted to Vertex AI.

## Local Development

Local CLI checks write to `/tmp` or `artifacts/` and are not tracked. The full local validation sequence is:

```bash
python -m pytest
python scripts/compile_pipeline.py --image-uri machledata:local
uvicorn apps.api:app --reload          # API on :8000
streamlit run apps/dashboard.py        # Dashboard on :8501
```
