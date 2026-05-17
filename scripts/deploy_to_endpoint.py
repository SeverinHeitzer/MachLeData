"""Deploy a model to a Vertex AI Endpoint."""

from __future__ import annotations

import argparse
from google.cloud import aiplatform

def main() -> None:
    """Upload a model and deploy it to a Vertex AI Endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", required=True, help="Google Cloud Project ID")
    parser.add_argument("--region", required=True, help="Vertex AI Region")
    parser.add_argument("--model-display-name", required=True, help="Display name for the uploaded model")
    parser.add_argument("--serving-image-uri", required=True, help="URI of the serving container image")
    parser.add_argument("--artifact-uri", required=True, help="URI of the model artifact (GCS path)")
    parser.add_argument("--endpoint-display-name", required=True, help="Display name for the endpoint")
    parser.add_argument("--machine-type", default="n1-standard-4", help="Machine type for deployment")
    
    args = parser.parse_args()

    print(f"Initializing Vertex AI SDK for project {args.project_id} in region {args.region}...")
    aiplatform.init(project=args.project_id, location=args.region)

    print(f"Uploading model '{args.model_display_name}'...")
    uploaded_model = aiplatform.Model.upload(
        display_name=args.model_display_name,
        serving_container_image_uri=args.serving_image_uri,
        artifact_uri=args.artifact_uri,
        serving_container_ports=[8080],
    )
    print(f"Model uploaded successfully: {uploaded_model.resource_name}")

    print(f"Creating endpoint '{args.endpoint_display_name}'...")
    endpoint = aiplatform.Endpoint.create(display_name=args.endpoint_display_name)
    print(f"Endpoint created successfully: {endpoint.resource_name}")

    print(f"Deploying model to endpoint using machine type {args.machine_type}...")
    deployed_model = uploaded_model.deploy(
        endpoint=endpoint,
        machine_type=args.machine_type,
        min_replica_count=1,
        max_replica_count=1,
    )
    
    print(f"Model deployed to endpoint successfully.")

if __name__ == "__main__":
    main()
