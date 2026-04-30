# MachLeData

MachLeData is a course project for a production-oriented YOLOv8 object detection pipeline with an MLOps layout. The repository targets Kubeflow Pipelines v2 and Vertex AI Pipelines for orchestration and uses Ultralytics YOLOv8 as the detection model.

## What Is In Place

- YOLOv8 inference via `machledata.infer.predict_image` — loads the model, runs detection, returns typed `Detection` objects
- YOLOv8 training via `machledata.train.train_yolo_model` with configurable epochs, batch size, and model variant
- Detection statistics and class-distribution metrics in `machledata.metrics`
- FastAPI service with `/health`, `/model/config`, and `/predict` endpoints
- Streamlit dashboard with Live Detection, Statistics, Batch Processing, and About modes
- Serializable prepare, train, evaluate, and publish orchestration helpers in `machledata.orchestration`
- Container-based Kubeflow Pipelines v2 definition in `workflows/kubeflow_pipeline.py`
- CLI scripts for training, prediction, evaluation, pipeline compilation, and Vertex submission
- Docker image running as a non-root user, with health endpoint exposed on port 8000
- Pytest coverage for data helpers, inference schema, API behavior, orchestration contracts, Kubeflow compilation, and Vertex submit wiring
- GitHub Actions CI/CD for tests, pipeline compilation, image build, push, and VPS deployment

BigQuery access has a normalized images-plus-labels contract but requires real GCP credentials and a populated dataset. Local sample images in `data/samples/` are used for offline smoke checks.

## Repository Layout

```
src/machledata/   reusable package — data, training, inference, metrics, orchestration, pipeline adapters
apps/             FastAPI (api.py) and Streamlit (dashboard.py) entry points
scripts/          CLI helpers: train, predict, evaluate, compile pipeline, submit to Vertex
configs/          committed YAML settings (model, data, app, pipeline)
tests/            pytest coverage for package, API, and pipeline behavior
workflows/        Kubeflow Pipelines v2 definition (kubeflow_pipeline.py)
docs/             architecture and data-source notes
docker/           Dockerfile and Compose configuration
data/samples/     tiny tracked fixtures only
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,kubeflow,vertex]"
```

Verify the installation:

```bash
python -m machledata
python -m pytest
```

## Common Commands

**API and dashboard:**

```bash
# FastAPI (http://localhost:8000)
uvicorn apps.api:app --reload

# Streamlit dashboard (http://localhost:8501)
streamlit run apps/dashboard.py
```

**CLI scripts:**

```bash
python scripts/train.py
python scripts/predict.py --image path/to/image.jpg
python scripts/evaluate.py
```

**Kubeflow pipeline:**

```bash
# Compile to YAML
python scripts/compile_pipeline.py --image-uri machledata:local

# Compile with Artifact Registry image for Vertex
python scripts/compile_pipeline.py \
  --image-uri "$VERTEX_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/machledata/machledata:latest"

# Submit to Vertex AI
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

**Docker Compose:**

```bash
docker compose -f docker/docker-compose.yml up api
```

## Kubeflow and Vertex Pipeline

The orchestration entry point is `workflows/kubeflow_pipeline.py`. It defines a single container-based Kubeflow Pipelines v2 workflow with four stages:

| Stage | Output artifacts |
|---|---|
| `prepare-data` | `system.Dataset` descriptor |
| `train-model` | `system.Model` + `system.Artifact` training metadata |
| `evaluate-model` | `system.Artifact` evaluation summary + `system.Metrics` |
| `publish-artifact-metadata` | `system.Artifact` serving manifest |

Each stage runs the shared Docker image and calls `python -m machledata.pipeline_steps <command>`. Artifacts are exchanged as KFP v2 typed paths with JSON payloads inside, keeping local smoke tests, Kubeflow, and Vertex AI aligned.

The `train-model` step currently writes a placeholder model artifact. Real training is implemented in `machledata.train.train_yolo_model` and requires a YOLO-format `dataset.yaml` to be generated during `prepare-data`. That wiring is pending.

`scripts/compile_pipeline.py` compiles the pipeline to `artifacts/pipelines/machledata_pipeline.yaml`. Compilation is local and requires no GCP credentials. Submission requires Google Cloud authentication and a `gs://` pipeline root the Vertex AI service account can write to.

Vertex submission parameters:

| Parameter | Default | Description |
|---|---|---|
| `dataset_id` | `local-demo` | Logical name for the prepared dataset |
| `samples_dir` | `data/samples` | Local image directory for offline smoke checks |
| `project_id` | `GOOGLE_CLOUD_PROJECT` | GCP project |
| `bigquery_dataset` | `BIGQUERY_DATASET` | BigQuery dataset name |
| `images_table` | `images` | Image metadata table |
| `labels_table` | `labels` | Bounding-box annotation table |
| `split` | `train` | Dataset split filter |
| `max_rows` | `0` (no limit) | Row cap for smoke runs |
| `model_name` | `yolov8n` | YOLO variant (`yolov8n`, `yolov8s`, …) |
| `epochs` | `10` | Training epochs |
| `artifact_root` | `VERTEX_ARTIFACT_ROOT` | Base path for run artifacts (`gs://…` on Vertex) |
| `run_label` | `manual` | Human-readable tag for the run |
| `min_detections_per_image` | `0.1` | Evaluation pass threshold |

## API Contract

```
GET  /health         → {"status": "ok"}
GET  /model/config   → {"model_name": ..., "image_size": ..., "confidence_threshold": ...}
POST /predict        → PredictionResponse
```

`POST /predict` accepts a multipart image upload and an optional `confidence_threshold` query parameter:

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

`bbox` is `[x1, y1, x2, y2]` in pixel coordinates. Returns an empty list when no objects are detected above the threshold.

## Configuration

YAML files in `configs/` provide committed defaults:

| File | Purpose |
|---|---|
| `configs/model.yaml` | Model variant, image size, batch size, epochs, artifact directory |
| `configs/data.yaml` | BigQuery dataset settings and local sample directory |
| `configs/app.yaml` | App title, model artifact path, confidence threshold |
| `configs/pipeline.yaml` | Kubeflow/Vertex defaults: image URI, pipeline root, region, artifact root |

Keep secrets and credentials out of Git. Use `.env` or deployment configuration for sensitive values. Required environment variables:

| Variable | Used by |
|---|---|
| `GOOGLE_CLOUD_PROJECT` | BigQuery, Vertex AI |
| `BIGQUERY_DATASET` | Data preparation step |
| `MODEL_ARTIFACT_PATH` | Inference service |
| `MACHLEDATA_PIPELINE_IMAGE` | Pipeline config default |
| `VERTEX_REGION` | Vertex AI submission and Artifact Registry |
| `VERTEX_PIPELINE_ROOT` | Vertex AI pipeline root (`gs://...`) |
| `VERTEX_SERVICE_ACCOUNT` | Vertex AI service account email |
| `VERTEX_ARTIFACT_ROOT` | GCS path for run artifacts (`gs://bucket/machledata-artifacts`) |

Passing `--image-uri` explicitly when compiling is preferred over relying on the environment variable so the compiled YAML contains a pinned image reference.

## CI/CD and Deployment

GitHub Actions runs on every push to `main` or `develop` and on pull requests:

1. **Test**: install `.[dev,kubeflow,vertex]`, run pytest, compile Kubeflow pipeline, upload coverage to Codecov
2. **Build**: build Docker image, tag with branch/semver/SHA, push to Docker Hub
3. **Deploy** (main only): SSH into VPS, pull new image, restart Compose stack, verify `/health`

Required GitHub secrets:

| Secret | Purpose |
|---|---|
| `DOCKER_USERNAME` | Docker Hub login |
| `DOCKER_PASSWORD` | Docker Hub token |
| `REMOTE_HOST` | VPS IP or hostname |
| `REMOTE_USER` | SSH user |
| `SSH_PRIVATE_KEY` | SSH private key |
| `REMOTE_APP_DIR` | Path to the app directory on the VPS |

## Development Notes

- Requires Python 3.11 or newer
- Keep package logic in `src/machledata/`; keep apps, scripts, and workflows thin
- `machledata.pipeline_steps` is the container boundary for Kubeflow steps — CLI args in, JSON artifact files out
- Kubeflow component interfaces use KFP v2 typed artifacts; JSON payloads inside artifact paths carry structured data between stages
- Do not commit generated artifacts, large datasets, caches, virtual environments, or trained model weights
- Add tests alongside new behavior, especially for inference, metrics, API routes, data helpers, and pipeline contracts

## Related Docs

- `docs/architecture.md` — pipeline structure and Vertex AI handoff
- `docs/data.md` — BigQuery schema and local fixture conventions
- `docs/gcp_setup.md` — one-time GCP setup: APIs, IAM, Artifact Registry, BigQuery tables, Workload Identity
- `data/README.md` — data directory layout
- `data/samples/README.md` — fixture files
