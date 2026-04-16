"""Command-line entry point for local image prediction.

This script should accept image paths, call `machledata.infer`, and print or
persist predictions in a format useful for demos and tests.
"""

from machledata.infer import predict_image


def main() -> None:
    """Run placeholder prediction on the sample directory path."""
    print(predict_image("data/samples"))


if __name__ == "__main__":
    main()

