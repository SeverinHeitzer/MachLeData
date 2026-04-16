# Shared development container for the MachLeData project.
# Split this file once API, dashboard, and training runtimes need different images.

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY apps ./apps

RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "machledata"]
