# Suggested Development Commands

## Environment Setup
```bash
# Install dependencies using uv
uv sync

# Activate virtual environment (if not using uv run)
source .venv/bin/activate

# Install pre-commit hooks
pre-commit install
```

## Running the Application
```bash
# Development server with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

# Or using uv
uv run uvicorn app.main:app --reload

# Production server
uvicorn app.main:app --host 0.0.0.0 --port 8080

# With Docker Compose (includes PostgreSQL and Redis)
docker-compose up -d
```

## Database Operations
```bash
# Create database migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Downgrade migration
alembic downgrade -1

# Reset database (development only)
alembic downgrade base && alembic upgrade head
```

## Code Quality & Testing
```bash
# Format code
black app/ tests/
# Or with uv
uv run black app/ tests/

# Sort imports
isort app/ tests/

# Lint code
ruff check app/ tests/

# Type checking
mypy app/

# Run all linting and formatting
pre-commit run --all-files

# Run tests
pytest
pytest -v  # verbose output
pytest --cov=app  # with coverage
pytest tests/test_specific.py  # specific test file

# Run tests with uv
uv run pytest
```

## Development Workflow
```bash
# Check git status and current branch
git status && git branch

# Create feature branch
git checkout -b feature/your-feature-name

# Run quality checks before commit
pre-commit run --all-files

# Commit changes
git add .
git commit -m "descriptive commit message"

# Push feature branch
git push -u origin feature/your-feature-name
```

## API Testing
```bash
# Start services
docker-compose up -d

# Test health endpoint
curl http://localhost:8080/health

# Test clean endpoint (requires API key)
curl -X POST "http://localhost:8080/v1/clean/" \
  -H "X-API-Key: test-api-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "1\n00:00:01,000 --> 00:00:03,000\nHello world",
    "format": "srt",
    "language": "en"
  }'

# Access API documentation (development mode)
open http://localhost:8080/docs
```

## Docker Operations
```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f api
docker-compose logs -f db
docker-compose logs -f redis

# Stop services
docker-compose down

# Remove volumes (reset data)
docker-compose down -v

# Shell into running container
docker-compose exec api bash
docker-compose exec db psql -U postgres -d subtitle_cleaner
```

## Monitoring & Debugging
```bash
# View application logs
tail -f logs/app.log

# Monitor metrics
curl http://localhost:8080/metrics

# Check Redis cache
docker-compose exec redis redis-cli
> keys *
> get "some-key"

# Check PostgreSQL database
docker-compose exec db psql -U postgres -d subtitle_cleaner
\dt  -- list tables
\d+ table_name  -- describe table
```

## Dependency Management
```bash
# Add new dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Update dependencies
uv sync

# Check for outdated packages
uv show --outdated

# Remove dependency
uv remove package-name
```

## System Utilities (macOS)
```bash
# File operations
ls -la  # list files with details
find . -name "*.py" -type f  # find Python files
grep -r "search_term" app/  # search in files

# Process management
ps aux | grep python  # find Python processes
kill -9 PID  # kill process by PID

# Network
netstat -an | grep 8080  # check if port is in use
lsof -i :8080  # what's using port 8080

# Git operations
git log --oneline -10  # last 10 commits
git diff HEAD~1  # diff with previous commit
git reset --hard HEAD  # reset to last commit (careful!)
```