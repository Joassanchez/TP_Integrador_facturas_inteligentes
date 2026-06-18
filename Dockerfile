# Multi-stage Dockerfile for invoice-pipeline execution
# Builder: installs Python dependencies (cached layer)
# Runtime: copies deps + source for minimal final image

FROM python:3.11-slim-bookworm AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy installed packages from builder (same Python version, same paths)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy requirements.txt for transparency (deps already installed via builder)
COPY requirements.txt .

# Copy application code (tests/ depends on data/ at runtime, not build time)
COPY src/ src/
COPY tests/ tests/
COPY data/ data/

# Default command: quality report pipeline (overridable via docker run / compose)
CMD ["python", "-m", "src.data.pipeline"]
