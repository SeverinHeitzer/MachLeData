# Shared development container for the MachLeData project.
# Split this file once API, dashboard, and training runtimes need different images.

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

ENV YOLO_CONFIG_DIR=/tmp/yolo_config

# Install system dependencies for OpenCV and PyTorch (OpenMP)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY apps ./apps
COPY configs ./configs
COPY scripts ./scripts

# Install package and pre-download default model weights for offline use
RUN pip install --no-cache-dir -e . \
    && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
    && useradd -m appuser \
    && mkdir -p /tmp/Ultralytics \
    && chmod 777 /tmp/Ultralytics
USER appuser

EXPOSE 8000

CMD ["uvicorn", "apps.api:app", "--host", "0.0.0.0", "--port", "8000"]
