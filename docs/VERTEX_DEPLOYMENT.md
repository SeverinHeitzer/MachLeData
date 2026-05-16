# Vertex AI Deployment — Handoff Notes

## Status

The MachLeData pipeline now runs end-to-end on Google Cloud Vertex AI:

- Cloud Build → Artifact Registry image
- Kubeflow 4-stage DAG: prepare-data → train-model → evaluate-model → publish-artifact-metadata
- Reads 128 images + 929 annotations from BigQuery
- Downloads images from GCS into the container, generates YOLO-format labels
- Uploads prepared dataset to GCS for cross-container handoff
- Real YOLOv8n training (~6 MB .pt artifact written to pipeline_root)
- All four DAG steps green on the latest run

## Quick verification

The most recent successful run is machledata-ml-pipeline-20260514183426. To see the trained model:

    gsutil ls -lh "gs://machledata-495608-machledata/pipeline_root/455909353739/machledata-ml-pipeline-20260514183426/train-model-component_*/model_artifact"

Should show a 5.99 MiB file.

## Setup to run the pipeline yourself

### Prerequisites

- gcloud CLI installed and authenticated (gcloud auth login and gcloud auth application-default login)
- Python 3.11+
- Git Bash (Windows) or any shell (Mac/Linux)
- Owner or Editor on the machledata-495608 GCP project

### One-time local setup

    git fetch origin
    git checkout feat/vertex-deployment

    python -m venv .venv
    source .venv/Scripts/activate    # Windows Git Bash
    # source .venv/bin/activate      # Mac/Linux

    pip install -e ".[dev,kubeflow,vertex]"

    cp .env.example .env             # Then edit .env if needed; defaults work for our project

### Every-session warmup

    cd path/to/MachLeData
    source .venv/Scripts/activate
    set -a; source .env; set +a

### Submit a pipeline run

    python scripts/submit_vertex_pipeline.py \
      --project-id "$GOOGLE_CLOUD_PROJECT" \
      --region "$VERTEX_REGION" \
      --pipeline-root "$VERTEX_PIPELINE_ROOT" \
      --artifact-root "//tmp/machledata-artifacts" \
      --template-path artifacts/pipelines/machledata_pipeline.yaml \
      --bigquery-dataset "$BIGQUERY_DATASET" \
      --run-label "your-label-here"

The double-slash in --artifact-root "//tmp/..." is intentional; it prevents Git Bash on Windows from translating the path to a Windows path before it reaches the container.

Watch the run at:
https://console.cloud.google.com/vertex-ai/locations/europe-west6/pipelines

### Rebuilding the Docker image (only after Python code changes)

    gcloud builds submit \
      --region="$VERTEX_REGION" \
      --config=cloudbuild.yaml \
      --substitutions=_IMAGE="$MACHLEDATA_PIPELINE_IMAGE" \
      --service-account="projects/$GOOGLE_CLOUD_PROJECT/serviceAccounts/$VERTEX_SERVICE_ACCOUNT" \
      .

Then recompile and re-submit. After recompile you must re-apply the caching patch:

    python scripts/compile_pipeline.py --image-uri "$MACHLEDATA_PIPELINE_IMAGE"
    sed -i 's/cachingOptions: {}/cachingOptions:\n          enableCache: false/g' artifacts/pipelines/machledata_pipeline.yaml

## Known caveats and limitations

1. The training set is small. Only 128 unique COCO128 images (929 labeled boxes). Training runs end-to-end and produces a real model, but the model won't be especially accurate. Increase --max-rows or load more images into BigQuery to improve.

2. Caching workaround. Vertex AI's per-component caching is hard to disable through the Python KFP SDK. We use a sed patch after each compile to force enableCache: false into the YAML. This must be re-applied after every compile_pipeline.py call.

3. One local test times out. tests/test_pipeline_steps.py::test_prepare_train_evaluate_publish_step_cli_round_trip now actually trains YOLO end-to-end locally and takes longer than the 30-second test timeout. It should be marked as @pytest.mark.integration or skipped on CI. Other 12 tests pass.

4. No GPU. europe-west6 doesn't offer NVIDIA T4. Training runs on CPU. The GPU branch in workflows/kubeflow_pipeline.py is commented out.

5. Cross-container data handoff via GCS. The pipeline uploads the prepared YOLO dataset to gs://machledata-495608-machledata/pipeline_artifacts/<run_label>/ and the train step downloads it. Proper KFP Output[Dataset] artifacts would be cleaner; this is a pragmatic workaround.

## Suggested next steps (not yet done)

### 1. Bigger training run (5 min of your time, ~30-45 min of pipeline time)

Submit with all 929 annotations and more epochs:

    python scripts/submit_vertex_pipeline.py \
      --project-id "$GOOGLE_CLOUD_PROJECT" \
      --region "$VERTEX_REGION" \
      --pipeline-root "$VERTEX_PIPELINE_ROOT" \
      --artifact-root "//tmp/machledata-artifacts-bigrun" \
      --template-path artifacts/pipelines/machledata_pipeline.yaml \
      --bigquery-dataset "$BIGQUERY_DATASET" \
      --run-label "full-30-epochs" \
      --epochs 30

No --max-rows flag means all rows.

### 2. Vertex AI Endpoint deployment (Phase E, ~2-3 hours)

Matches lecture slides 56-61. Add either:

- A deploy-model-component to the Kubeflow pipeline, OR
- A standalone scripts/deploy_to_endpoint.py script

Both call aiplatform.Model.upload(...) and endpoint.deploy(...). The standalone script is faster to ship. After deployment, the model is queryable via the Vertex Predict API.

### 3. Fix the failing test

Either skip it locally with @pytest.mark.skip or mark it as integration-only.

### 4. Update READMEs and architecture notes

Add a short section to the main README describing the deployment architecture and pointing to this doc.

## Useful Cloud Console links

- Vertex AI Pipelines: https://console.cloud.google.com/vertex-ai/locations/europe-west6/pipelines
- Artifact Registry: https://console.cloud.google.com/artifacts/docker/machledata-495608/europe-west6/machledata
- GCS bucket: https://console.cloud.google.com/storage/browser/machledata-495608-machledata
- BigQuery: https://console.cloud.google.com/bigquery?project=machledata-495608
- Latest pipeline run: https://console.cloud.google.com/vertex-ai/locations/europe-west6/pipelines/runs/machledata-ml-pipeline-20260514183426?project=machledata-495608

## Questions?

The PR with all changes is at feat/vertex-deployment. Each commit message has context. Logs from each pipeline run are in Cloud Logging — filter by resource.type="ml_job" and the specific job_id.
