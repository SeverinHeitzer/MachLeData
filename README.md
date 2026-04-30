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

The real YOLO training, BigQuery/image dataset integration, and production inference logic are still placeholders. After this scaffold, the main remaining work is to wire the chosen model and dataset into the existing seams.

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

Submit the compiled pipeline to Vertex AI:

```bash
python scripts/submit_vertex_pipeline.py \
  --project-id "$GOOGLE_CLOUD_PROJECT" \
  --region "$VERTEX_REGION" \
  --pipeline-root "$VERTEX_PIPELINE_ROOT" \
  --template-path artifacts/pipelines/machledata_pipeline.yaml
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

Each component runs the shared project image and calls `python -m machledata.pipeline_steps ...`. Pipeline stages exchange JSON files so the real model can be introduced without changing the orchestration shape:

- `prepared_dataset.json`
- `training_run.json`
- `evaluation_summary.json`
- `artifact_manifest.json`

`scripts/compile_pipeline.py` compiles the pipeline to `artifacts/pipelines/machledata_pipeline.yaml`. `scripts/submit_vertex_pipeline.py` submits that package to Vertex AI Pipelines with explicit GCP settings. Nothing is submitted to GCP automatically during tests.

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

## CI/CD and Deployment

The GitHub Actions workflow runs tests, compiles the Kubeflow pipeline, builds the Docker image, pushes it to Docker Hub, and deploys the API stack to a remote VPS on pushes to `main`.

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
- Avoid committing generated artifacts, large datasets, caches, virtual environments, or trained model weights
- Prefer adding tests alongside new behavior, especially for inference, metrics, API routes, data helpers, and pipeline contracts

## Related Docs

- `docs/architecture.md`
- `docs/data.md`
- `data/README.md`
- `data/samples/README.md`
