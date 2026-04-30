# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (requires Python 3.11+)
pip install -e ".[dev,kubeflow,vertex]"

# Tests
.venv/bin/pytest                          # full suite
.venv/bin/pytest tests/test_api.py -v     # single file
.venv/bin/pytest -k test_health -v        # single test

# Run services
uvicorn apps.api:app --reload             # API on :8000
streamlit run apps/dashboard.py           # dashboard on :8501

# Pipeline
python scripts/compile_pipeline.py --image-uri machledata:local
python scripts/submit_vertex_pipeline.py --project-id "$GOOGLE_CLOUD_PROJECT" \
  --region "$VERTEX_REGION" --pipeline-root "$VERTEX_PIPELINE_ROOT" \
  --template-path artifacts/pipelines/machledata_pipeline.yaml

# Docker
docker build -f docker/Dockerfile -t machledata:local .
docker compose -f docker/docker-compose.yml up api
```

## Architecture

### Layer boundaries

```
src/machledata/      ← all ML logic lives here (importable package)
  data.py            BigQuery client + local sample loader
  model.py           YOLO model load/save wrappers
  train.py           train_yolo_model() — wraps ultralytics.YOLO.train()
  infer.py           predict_image(), predict_batch(); Detection + PredictionResponse schemas
  metrics.py         detection statistics, class distribution, evaluate_on_images()
  orchestration.py   four pipeline functions: prepare_dataset → train_model → evaluate_model → publish_artifact_manifest
  pipeline_steps.py  CLI adapter — translates argparse args/KFP artifact paths to orchestration calls

apps/                ← thin consumers, no ML logic
  api.py             FastAPI: /health, /model/config, POST /predict
  dashboard.py       Streamlit: Live Detection, Statistics, Batch, About modes

scripts/             ← CLI entry points, no business logic
  train.py / predict.py / evaluate.py   local smoke-test runners
  compile_pipeline.py                   kfp.compiler → YAML
  submit_vertex_pipeline.py             google-cloud-aiplatform → PipelineJob

workflows/
  kubeflow_pipeline.py   @dsl.container_component definitions + @dsl.pipeline graph
```

### How Kubeflow pipeline data flows

Each stage runs the Docker image and calls `python -m machledata.pipeline_steps <command>`. KFP v2 typed artifacts (`dsl.Dataset`, `dsl.Model`, `dsl.Artifact`, `dsl.Metrics`) carry paths; JSON payloads are written inside those paths.

```
prepare-data   → dataset_descriptor.json  (dsl.Dataset)
train-model    → training_run.json + model artifact  (dsl.Model + dsl.Artifact)
evaluate-model → evaluation_summary.json + metrics  (dsl.Artifact + dsl.Metrics)
publish        → artifact_manifest.json  (dsl.Artifact, only when eval passes)
```

`orchestration.py` is the single source of truth for step logic. `pipeline_steps.py` is only a CLI boundary — it reads args, calls the matching `orchestration.*` function, and writes outputs to the KFP-provided paths.

### Detection schema (shared across API and dashboard)

`machledata.infer.Detection` — `class_name: str`, `confidence: float`, `bbox: tuple[float, float, float, float]` (x1, y1, x2, y2 pixels). `PredictionResponse` wraps a list of these. Both `api.py` and `dashboard.py` import directly from `machledata.infer`.

### Configuration

YAML files in `configs/` are loaded by `orchestration.load_yaml_config()` which expands `${ENV_VAR}` placeholders. `config_value()` strips unresolved placeholders to `None`. Never embed credentials in these files.

### Model cache

`infer.py` holds a module-level `_model_cache` dict protected by `threading.Lock`. Models are loaded once per process. Call `clear_model_cache()` to free memory between runs.

## GCP setup

One-time GCP setup (APIs, service account, IAM, Artifact Registry, BigQuery tables, Workload Identity Federation) is documented in `docs/gcp_setup.md`.

## Key constraints

- `machledata.pipeline_steps` is the container boundary — keep it thin, push logic into `machledata.orchestration`
- KFP component interfaces must stay typed (`dsl.Input[dsl.Dataset]` etc.) for Vertex artifact lineage
- `artifact_root` must be a `gs://` path for real Vertex runs; `/tmp` only works locally
- The Docker image is shared between the API service and all Kubeflow pipeline step containers
- Optional extras: `.[kubeflow]` adds kfp, `.[vertex]` adds google-cloud-aiplatform — not in the base image by default
