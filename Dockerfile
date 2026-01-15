# Base image with uv
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install the package using uv
RUN uv sync --no-dev

# Create data directory
RUN mkdir -p /data

# Expose port
EXPOSE 8000

# Set environment variables for the app
ENV DATABASE_PATH=/data/museum_data.db \
    MIN_VISITORS=2000000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Run the API server
CMD ["uv", "run", "uvicorn", "museum_regression.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
