"""Package smoke-test entry point.

Running `python -m machledata` confirms the package can be imported in local
development and container images.
"""


def main() -> None:
    """Print a short package readiness message."""
    print("machledata package is installed")


if __name__ == "__main__":
    main()

