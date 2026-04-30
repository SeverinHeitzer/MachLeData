"""Command-line entry point for local image prediction.

This script accepts image paths, calls `machledata.infer`, and prints
predictions in a format useful for demos and tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from machledata.data import load_sample_paths
from machledata.infer import predict_batch, predict_image
from machledata.model import build_model_config


def main() -> None:
    """Run object detection on provided image paths."""
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on images",
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default="data/samples",
        help="Path to image file or directory (default: data/samples)",
    )
    parser.add_argument(
        "--model-name",
        default="yolov8n",
        help="YOLO model to use (default: yolov8n)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to saved model file (optional)",
    )
    args = parser.parse_args()

    # Build config
    config = build_model_config(
        model_name=args.model_name,
        image_size=args.image_size,
        confidence_threshold=args.confidence,
    )

    # Resolve image paths
    path = Path(args.image_path)
    image_paths = []

    if path.is_file():
        image_paths = [path]
    elif path.is_dir():
        # Get sample images from directory
        image_paths = load_sample_paths(path)
    else:
        # Try default samples
        image_paths = load_sample_paths("data/samples")

    if not image_paths:
        print(json.dumps({"error": "No images found", "path": str(path)}, indent=2))
        return

    # Run predictions
    if len(image_paths) == 1:
        detections = predict_image(image_paths[0], args.model_path, config)
        result = {
            "image": str(image_paths[0]),
            "detection_count": len(detections),
            "detections": [
                {
                    "class_name": d.class_name,
                    "confidence": float(d.confidence),
                    "bbox": {
                        "x1": float(d.bbox[0]),
                        "y1": float(d.bbox[1]),
                        "x2": float(d.bbox[2]),
                        "y2": float(d.bbox[3]),
                    },
                }
                for d in detections
            ],
        }
    else:
        predictions = predict_batch(image_paths, args.model_path, config)
        result = {
            "image_count": len(image_paths),
            "total_detections": sum(len(d) for d in predictions.values()),
            "predictions": {
                str(img_path): [
                    {
                        "class_name": d.class_name,
                        "confidence": float(d.confidence),
                        "bbox": {
                            "x1": float(d.bbox[0]),
                            "y1": float(d.bbox[1]),
                            "x2": float(d.bbox[2]),
                            "y2": float(d.bbox[3]),
                        },
                    }
                    for d in predictions[str(img_path)]
                ]
                for img_path in image_paths
            },
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

