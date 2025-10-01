# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI application for SRT (subtitle) file processing. The project uses Python 3.13 with the `uv` package manager for dependency management.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Running the Application
```bash
# Development server with auto-reload
uvicorn main:app --reload

# Production server
uvicorn main:app
```

### Dependency Management
```bash
# Add a new dependency
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>

# Update dependencies
uv sync
```

## Architecture

The application is built with FastAPI and currently has a minimal structure:
- `main.py`: Entry point containing the FastAPI application instance and route definitions
- Uses Python 3.13 as specified in `.python-version`
- Dependencies managed via `uv` with lock file (`uv.lock`) for reproducible builds