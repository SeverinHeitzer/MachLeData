"""Command-line entry point for local image prediction.

This script should accept image paths, call `machledata.infer`, and print or
persist predictions in a format useful for demos and tests.
"""

import json

from machledata.infer import PredictionResponse, predict_image


def main() -> None:
    """Run placeholder prediction on the sample directory path."""
    response = PredictionResponse(detections=predict_image("data/samples"))
    print(json.dumps(response.model_dump(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
