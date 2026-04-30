# Shared development container for the MachLeData project.
# Split this file once API, dashboard, and training runtimes need different images.

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY apps ./apps
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --no-cache-dir -e . \
    && useradd -m appuser

USER appuser

EXPOSE 8000

CMD ["uvicorn", "apps.api:app", "--host", "0.0.0.0", "--port", "8000"]
