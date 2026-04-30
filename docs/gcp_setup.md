# GCP Setup Guide

One-time setup steps to run the MachLeData pipeline on Vertex AI.

## Prerequisites

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and configured
- A GCP project with billing enabled
- Owner or Editor role on the project (to grant IAM roles)

---

## 1. Enable APIs

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  --project "$GOOGLE_CLOUD_PROJECT"
```

---

## 2. Create a Service Account

```bash
gcloud iam service-accounts create machledata-pipeline \
  --display-name "MachLeData Pipeline SA" \
  --project "$GOOGLE_CLOUD_PROJECT"

SA_EMAIL="machledata-pipeline@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com"
```

### Required IAM roles

```bash
for ROLE in \
  roles/aiplatform.user \
  roles/bigquery.dataViewer \
  roles/bigquery.jobUser \
  roles/storage.objectAdmin \
  roles/artifactregistry.reader \
  roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$ROLE"
done
```

---

## 3. Create GCS Bucket (artifact root)

```bash
gcloud storage buckets create "gs://${GOOGLE_CLOUD_PROJECT}-machledata" \
  --location "$VERTEX_REGION" \
  --project "$GOOGLE_CLOUD_PROJECT"

# Set VERTEX_ARTIFACT_ROOT in your .env:
# VERTEX_ARTIFACT_ROOT=gs://${GOOGLE_CLOUD_PROJECT}-machledata/artifacts
```

---

## 4. Create Artifact Registry Repository

```bash
gcloud artifacts repositories create machledata \
  --repository-format docker \
  --location "$VERTEX_REGION" \
  --project "$GOOGLE_CLOUD_PROJECT"
```

The pipeline image URI will be:

```
${VERTEX_REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/machledata/machledata:latest
```

---

## 5. Configure GitHub Actions (Workload Identity Federation)

Allows GitHub Actions to push images and authenticate to GCP without storing a service account key.

```bash
# Create a Workload Identity Pool
gcloud iam workload-identity-pools create github-pool \
  --location global \
  --project "$GOOGLE_CLOUD_PROJECT"

POOL_ID=$(gcloud iam workload-identity-pools describe github-pool \
  --location global \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --format "value(name)")

# Create a provider for GitHub
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location global \
  --workload-identity-pool github-pool \
  --issuer-uri "https://token.actions.githubusercontent.com" \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --project "$GOOGLE_CLOUD_PROJECT"

PROVIDER_ID=$(gcloud iam workload-identity-pools providers describe github-provider \
  --location global \
  --workload-identity-pool github-pool \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --format "value(name)")

# Allow the GitHub repo to impersonate the service account
# Replace SeverinHeitzer/MachLeData with your repo
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/SeverinHeitzer/MachLeData" \
  --project "$GOOGLE_CLOUD_PROJECT"
```

Add these as GitHub repository secrets:

| Secret | Value |
|---|---|
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | output of `echo $PROVIDER_ID` |
| `GCP_SERVICE_ACCOUNT` | `machledata-pipeline@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com` |
| `GOOGLE_CLOUD_PROJECT` | your project ID |
| `VERTEX_REGION` | e.g. `europe-west4` |

---

## 6. BigQuery Tables

```sql
CREATE TABLE `your_project.your_dataset.images` (
  image_id   STRING  NOT NULL,
  image_uri  STRING  NOT NULL,  -- gs:// path to image file
  split      STRING  NOT NULL,  -- "train", "val", or "test"
  width      INT64,
  height     INT64
);

CREATE TABLE `your_project.your_dataset.labels` (
  image_id   STRING  NOT NULL,
  class_name STRING,
  x_min      FLOAT64,
  y_min      FLOAT64,
  x_max      FLOAT64,
  y_max      FLOAT64
);
```

Set in your `.env`:

```
BIGQUERY_DATASET=your_dataset
```

> **Note**: Image URIs must be `gs://` paths accessible by the pipeline service account.
> The pipeline reads metadata from BigQuery but does **not** download images automatically.
> For Vertex AI training with real images, add a download step to `prepare_dataset()`
> that fetches images from GCS to a local temp directory before generating `dataset.yaml`.

---

## 7. Local Development Authentication

```bash
gcloud auth application-default login
gcloud config set project "$GOOGLE_CLOUD_PROJECT"
```

---

## 8. First Vertex Run

```bash
# 1. Build and push the pipeline image
docker build -f docker/Dockerfile -t \
  "${VERTEX_REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/machledata/machledata:latest" .
docker push \
  "${VERTEX_REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/machledata/machledata:latest"

# 2. Compile the pipeline with the Artifact Registry image
python scripts/compile_pipeline.py \
  --image-uri "${VERTEX_REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/machledata/machledata:latest"

# 3. Submit to Vertex AI (add --use-gpu for GPU training)
python scripts/submit_vertex_pipeline.py \
  --project-id "$GOOGLE_CLOUD_PROJECT" \
  --region "$VERTEX_REGION" \
  --pipeline-root "$VERTEX_PIPELINE_ROOT" \
  --artifact-root "$VERTEX_ARTIFACT_ROOT" \
  --template-path artifacts/pipelines/machledata_pipeline.yaml \
  --bigquery-dataset "$BIGQUERY_DATASET"
```
