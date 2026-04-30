# MachLeData

MachLeData is a course project for a YOLO-style object detection pipeline with an MLOps-oriented layout. The repository now targets Kubeflow Pipelines v2 and Vertex AI Pipelines for orchestration, while keeping the model implementation as a clear future integration point.

## Current Status

The project is a working MLOps scaffold. These pieces are in place:

- Python package entry point in `src/machledata/`
- FastAPI health and placeholder prediction endpoints in `apps/api.py`
- Streamlit dashboard that consumes the API prediction contract
- Local CLI scripts for training, prediction, evaluation, pipeline compilation, and Vertex submission
- Serializable prepare, train, evaluate, and publish helpers in `machledata.orchestration`
- Container-based Kubeflow Pipelines v2 definition in `workflows/kubeflow_pipeline.py`
- Docker image for API and pipeline step runtime
- Pytest coverage for data helpers, inference, API behavior, orchestration contracts, Kubeflow compilation, and Vertex submit wiring
- GitHub Actions CI/CD for tests, pipeline compilation, image build, and deployment automation

The real YOLO training and production inference logic are still placeholders. BigQuery access now has a normalized images-plus-labels contract, but the real dataset and credentials still need to be supplied from GCP. After this scaffold, the main remaining work is to wire the chosen model and dataset into the existing seams:

- `machledata.data.BigQueryDatasetConfig` and `machledata.data.load_bigquery_object_detection_rows` for dataset access
- `machledata.train.create_training_run` / `machledata.orchestration.train_model` for model training
- `machledata.infer.predict_image` for production prediction
- `machledata.metrics` and `machledata.orchestration.evaluate_model` for real evaluation metrics

## Repository Layout

- `src/machledata/`: reusable package code for data access, training, inference, metrics, orchestration, and pipeline step adapters
- `apps/`: FastAPI and Streamlit entry points
- `scripts/`: local command-line helpers, including Kubeflow compile and Vertex submit scripts
- `configs/`: committed YAML configuration files
- `tests/`: pytest coverage for package, API, and pipeline behavior
- `workflows/`: Kubeflow pipeline definitions
- `docs/`: architecture and data notes
- `docker/`: Dockerfile and Compose configuration for local services
- `data/samples/`: tiny tracked fixtures only

## Quick Start

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install development and pipeline dependencies:

```bash
python -m pip install -e ".[dev,kubeflow,vertex]"
```

Run the package smoke test:

```bash
python -m machledata
```

Run the test suite:

```bash
python -m pytest
```

Run the same validation shape used for pipeline work:

```bash
python -m pytest
python scripts/compile_pipeline.py --image-uri machledata:local --package-path /tmp/machledata_pipeline.yaml
```

## Common Commands

Run the FastAPI app locally:

```bash
python -m uvicorn apps.api:app --reload
```

Run the Streamlit dashboard:

```bash
streamlit run apps/dashboard.py
```

Smoke-test the CLI scripts:

```bash
python scripts/train.py
python scripts/predict.py
python scripts/evaluate.py
```

Compile the Kubeflow pipeline package:

```bash
python scripts/compile_pipeline.py --image-uri machledata:local
```

For Vertex AI, use an image URI that Vertex can pull, for example an Artifact Registry image:

```bash
python scripts/compile_pipeline.py \
  --image-uri "$VERTEX_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/machledata/machledata:latest"
```

Submit the compiled pipeline to Vertex AI:

```bash
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

Run the API service through Docker Compose:

```bash
docker compose -f docker/docker-compose.yml up api
```

## Kubeflow and Vertex Pipeline

The orchestration entry point is `workflows/kubeflow_pipeline.py`. It defines one container-based Kubeflow Pipelines v2 workflow:

1. `prepare-data`
2. `train-model`
3. `evaluate-model`
4. `publish-artifact-metadata`

Each component runs the shared project image and calls `python -m machledata.pipeline_steps ...`. Pipeline stages exchange Vertex-compatible Kubeflow Pipelines v2 artifacts so the real model can be introduced behind the package seams:

- `prepared_dataset`: `system.Dataset` descriptor, with optional sibling `.annotations.jsonl`
- `model_artifact`: `system.Model` placeholder path ready to become the trained model artifact
- `training_metadata`: `system.Artifact` JSON metadata for the run
- `evaluation_summary`: `system.Artifact` JSON summary
- `evaluation_metrics`: `system.Metrics` JSON metrics payload
- `artifact_manifest`: `system.Artifact` serving handoff manifest

`scripts/compile_pipeline.py` compiles the pipeline to `artifacts/pipelines/machledata_pipeline.yaml`. `scripts/submit_vertex_pipeline.py` submits that package to Vertex AI Pipelines with explicit GCP settings. Nothing is submitted to GCP automatically during tests.

The compile step is local and does not require GCP credentials. The submit step requires Google Cloud authentication and a `gs://` pipeline root that the Vertex AI pipeline service account can write to.

Vertex submission accepts the dataset path as separate BigQuery parameters:

- `project_id`: GCP project, normally `GOOGLE_CLOUD_PROJECT`
- `bigquery_dataset`: dataset name, normally `BIGQUERY_DATASET`
- `images_table`: image metadata table, default `images`
- `labels_table`: bounding-box label table, default `labels`
- `split`: dataset split filter, default `train`
- `max_rows`: optional smoke-test row cap; `0` means no limit

## API and Dashboard Contract

The FastAPI app exposes:

- `GET /health`: returns `{"status": "ok"}`
- `POST /predict`: accepts an uploaded image and returns:

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

The endpoint currently returns an empty detection list until real inference is implemented. The dashboard already consumes this response shape.

## Configuration

Committed YAML files in `configs/` define the initial project shape:

- `configs/model.yaml`: model name, image size, batch size, epochs, artifact directory
- `configs/data.yaml`: BigQuery-oriented dataset settings and sample directory
- `configs/app.yaml`: app title, model artifact path, and confidence threshold
- `configs/pipeline.yaml`: Kubeflow/Vertex defaults such as image URI, pipeline root, region, and artifact root

Keep secrets and environment-specific values out of Git. Use `.env` or deployment configuration for credentials and sensitive settings.

Relevant environment variables:

- `GOOGLE_CLOUD_PROJECT`
- `BIGQUERY_DATASET`
- `MODEL_ARTIFACT_PATH`
- `MACHLEDATA_PIPELINE_IMAGE`
- `VERTEX_REGION`
- `VERTEX_PIPELINE_ROOT`
- `VERTEX_SERVICE_ACCOUNT`

`MACHLEDATA_PIPELINE_IMAGE` is used by config defaults, but the recommended release workflow is to pass `--image-uri` explicitly when compiling the pipeline so the YAML contains the exact image tag intended for Vertex.

`configs/data.yaml` documents the committed BigQuery table contract: an `images` table with image IDs, URIs, split, width, and height, plus a `labels` table with image IDs, class names, and pixel-space `x_min`, `y_min`, `x_max`, `y_max` boxes. Credentials and private dataset exports stay outside the repository.

## CI/CD and Deployment

The GitHub Actions workflow runs tests, compiles the Kubeflow pipeline, builds the Docker image, pushes it to Docker Hub, and deploys the API stack to a remote VPS on pushes to `main`.

CI compiles the pipeline with a neutral `machledata:ci` image to validate syntax and task wiring. Production Vertex runs should be compiled with a concrete registry image that exists and is accessible from GCP.

The deployment workflow expects these repository secrets:

- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`
- `REMOTE_HOST`
- `REMOTE_USER`
- `SSH_PRIVATE_KEY`
- `REMOTE_APP_DIR`

## Development Notes

- Target Python 3.10 or newer
- Keep package logic in `src/machledata/`; leave apps, scripts, and workflows thin
- Use `machledata.pipeline_steps` as the container boundary for Kubeflow steps
- Keep Kubeflow component interfaces typed with KFP v2 artifacts and use JSON payloads inside those artifact paths for the current scaffold
- Keep FastAPI tests on the pinned `fastapi` / `starlette` / `httpx` compatibility range in `pyproject.toml`; `pytest-timeout` catches future TestClient hangs quickly
- Avoid committing generated artifacts, large datasets, caches, virtual environments, or trained model weights
- Prefer adding tests alongside new behavior, especially for inference, metrics, API routes, data helpers, and pipeline contracts

## Related Docs

- `docs/architecture.md`
- `docs/data.md`
- `data/README.md`
- `data/samples/README.md`
