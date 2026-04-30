# Data Directory

This directory is for small local samples and documentation about data access.
The project targets BigQuery and Google Cloud Storage style data sources for the
real object detection workflow, so large raw datasets, processed datasets, and
trained model artifacts should stay outside Git.

Use `data/samples/` for tiny images or fixtures that make tests and demos
reproducible without cloud access.

Generated pipeline outputs belong under `artifacts/` or a Vertex AI `gs://`
pipeline root, not in this directory.
