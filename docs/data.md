# Data Source Notes

MachLeData is prepared for a BigQuery-oriented object detection dataset, with
small local fixtures used only for tests and offline smoke checks.

## Current Contract

`configs/data.yaml` defines the committed data settings:

- `source`: currently `bigquery`
- `project_id`: read from `GOOGLE_CLOUD_PROJECT`
- `dataset`: read from `BIGQUERY_DATASET`
- `samples_dir`: local tiny fixture directory, defaulting to `data/samples`

`machledata.data.load_sample_paths` only includes image-like files from the
sample directory. Markdown notes or other metadata files are ignored so local
smoke tests do not accidentally count documentation as training data.

## Intended BigQuery Shape

The concrete schema is still to be finalized with the real dataset. The future
loader should provide enough information for a YOLO-style training step:

- stable image identifier
- image URI or storage path
- object class label
- bounding box coordinates
- split or filtering metadata when available

The pipeline should keep secrets and credentials outside Git. Use environment
variables, GCP service accounts, and Vertex AI pipeline service account
permissions rather than committed credentials.

## Local Fixtures

Use `data/samples/` only for tiny, non-sensitive image files that make tests or
demo smoke checks reproducible without cloud access. Do not store raw exports,
large datasets, trained weights, or generated artifacts in Git.

## Model Integration Boundary

When the real model is added, data loading should be implemented behind
`machledata.data` and consumed through `machledata.orchestration.prepare_dataset`.
Kubeflow and Vertex pipeline definitions should continue to call the same
package-level contract instead of embedding dataset logic in pipeline code.
