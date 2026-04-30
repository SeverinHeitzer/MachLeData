# Data Source Notes

MachLeData is prepared for a BigQuery-oriented object detection dataset, with
small local fixtures used only for tests and offline smoke checks.

## Current Contract

`configs/data.yaml` defines the committed data settings:

- `source`: currently `bigquery`
- `project_id`: read from `GOOGLE_CLOUD_PROJECT`
- `dataset`: read from `BIGQUERY_DATASET`
- `samples_dir`: local tiny fixture directory, defaulting to `data/samples`
- `images_table`: image metadata table, defaulting to `images`
- `labels_table`: bounding-box annotation table, defaulting to `labels`
- `split`: split filter for pipeline runs, defaulting to `train`

`machledata.data.load_sample_paths` only includes image-like files from the
sample directory. Markdown notes or other metadata files are ignored so local
smoke tests do not accidentally count documentation as training data.

`machledata.data.load_bigquery_object_detection_rows` is the cloud data seam.
It uses `google-cloud-bigquery` and returns normalized records with
`image_id`, `image_uri`, `split`, `width`, `height`, `class_name`, and `bbox`.

## BigQuery Shape

The prepared BigQuery schema is intentionally simple and YOLO-friendly:

`images` table:

- `image_id`: stable image identifier
- `image_uri`: image URI or storage path, such as `gs://...`
- `split`: train/validation/test split label
- `width`: source image width in pixels
- `height`: source image height in pixels

`labels` table:

- `image_id`: foreign key to `images.image_id`
- `class_name`: object class label
- `x_min`, `y_min`, `x_max`, `y_max`: pixel-space bounding box coordinates

The pipeline joins these tables on `image_id`, filters by `split`, and can use
`max_rows` for cheap smoke runs. `configs/data.yaml` can rename the table and
column names if the real dataset uses different identifiers.

The pipeline should keep secrets and credentials outside Git. Use environment
variables, GCP service accounts, and Vertex AI pipeline service account
permissions rather than committed credentials.

## Local Fixtures

Use `data/samples/` only for tiny, non-sensitive image files that make tests or
demo smoke checks reproducible without cloud access. Do not store raw exports,
large datasets, trained weights, or generated artifacts in Git.

## Model Integration

The YOLOv8 model is integrated via `machledata.train.train_yolo_model`. It expects a
`dataset.yaml` in the Ultralytics format pointing at an image directory and class list.

`machledata.orchestration.prepare_dataset` produces the dataset descriptor from which
the training step derives its inputs. Real end-to-end training requires:

1. Images accessible at the paths recorded in BigQuery (`image_uri` column, typically `gs://…`).
2. A YOLO-format `dataset.yaml` generated during data preparation and passed to `train_yolo_model(dataset_path=…)`.

Local smoke checks use `data/samples/` fixtures. Any additional image downloading or
GCS-to-local staging belongs behind `machledata.data` so the pipeline steps and CLI
scripts stay aligned.
