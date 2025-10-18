# Technology Stack

## Core Framework
- **Python 3.13**: Latest Python version
- **FastAPI**: Modern async web framework with automatic OpenAPI docs
- **Pydantic**: Data validation and settings management
- **SQLModel**: Type-safe database ORM built on SQLAlchemy

## Dependencies Management
- **uv**: Modern Python package manager for faster dependency resolution
- **pyproject.toml**: Project configuration and dependency specification

## Database & Caching
- **PostgreSQL 16**: Primary database for tenant memory and glossaries
- **Redis 7**: Caching and rate limiting
- **Alembic**: Database migrations
- **asyncpg**: Async PostgreSQL driver

## ML & NLP Libraries
- **Transformers**: Hugging Face models for punctuation, grammar correction
- **Sentence Transformers**: Semantic embeddings
- **spaCy**: NLP pipeline for entity recognition
- **OpenAI**: Optional LLM integration for disambiguation
- **ONNX Runtime**: Optimized model inference
- **PyTorch**: Deep learning framework

## Subtitle Processing
- **pysubs2**: SRT/WebVTT parsing and serialization
- **srt**: Additional subtitle format support

## Infrastructure
- **Docker**: Containerization with multi-service setup
- **Prometheus**: Metrics collection
- **OpenTelemetry**: Distributed tracing
- **Redis**: Rate limiting and caching

## Development Tools
- **Black**: Code formatting
- **Ruff**: Fast Python linting
- **isort**: Import sorting
- **mypy**: Static type checking
- **pytest**: Testing framework with async support
- **pre-commit**: Git hooks for code quality