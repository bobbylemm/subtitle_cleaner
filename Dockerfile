FROM python:3.11-slim

WORKDIR /app

# Cache bust argument to force rebuilds when needed
ARG CACHE_BUST=1

# Install system dependencies and UV in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  g++ \
  gcc \
  postgresql-client \
  && rm -rf /var/lib/apt/lists/* \
  && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

# Copy only dependency files first for better caching
COPY pyproject.toml uv.lock README.md ./

# Install dependencies with UV sync (this is cached unless pyproject.toml/uv.lock changes)
RUN uv sync --frozen --no-dev

# Copy application code (this layer changes frequently)
COPY app app/
COPY alembic alembic/
COPY alembic.ini alembic.ini

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Use PORT environment variable with default of 8080
ENV PORT=8080
EXPOSE ${PORT}

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Use shell form to allow environment variable expansion
CMD uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --log-level info