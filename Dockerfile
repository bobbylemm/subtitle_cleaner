FROM python:3.13-slim

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

# Download spaCy models for NLP tasks
# Install spacy models from GitHub releases using uv pip with --python flag to target the venv
RUN uv pip install --python /app/.venv/bin/python \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl \
    https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-3.8.0/xx_ent_wiki_sm-3.8.0-py3-none-any.whl

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
# Use the venv Python directly instead of uv run to avoid startup issues
# Enable --reload for hot reload during development (watches mounted volumes)
CMD /app/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --log-level info --reload