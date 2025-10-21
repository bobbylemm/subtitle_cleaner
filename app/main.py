import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.api.routers import clean, glossaries, health, preview, validate
from app.core.config import settings
from app.infra.cache import redis_client
from app.infra.db import close_db, init_db
from app.infra.metrics import (
    REQUEST_COUNT,
    REQUEST_DURATION,
    setup_metrics,
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Manage application lifecycle"""
    # Startup
    await init_db()
    await redis_client.initialize()
    setup_metrics()
    
    # Initialize ML models for scalable pipeline
    try:
        from app.services.pipeline_integration import initialize_models
        initialize_models()
    except Exception as e:
        print(f"Warning: Could not initialize ML models: {e}")
    
    yield
    
    # Shutdown
    await close_db()
    await redis_client.close()
    
    # Cleanup ML models
    try:
        from app.services.pipeline_integration import cleanup_models
        cleanup_models()
    except Exception:
        pass


app = FastAPI(
    title=settings.APP_NAME,
    description="API for cleaning and perfecting SRT/WebVTT subtitles across languages",
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "clean", "description": "Clean and process subtitle files"},
        {"name": "validate", "description": "Validate subtitle files"},
        {"name": "preview", "description": "Preview cleaning results"},
        {"name": "glossaries", "description": "Manage glossary terms"},
        {"name": "health", "description": "Health and monitoring endpoints"},
    ],
)

# Middleware
app.add_middleware(RequestIDMiddleware)

# CORS configuration
if settings.CORS_ORIGINS:
    origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(
    clean.router,
    prefix=f"{settings.API_PREFIX}/clean",
    tags=["clean"],
)
app.include_router(
    validate.router,
    prefix=f"{settings.API_PREFIX}/validate",
    tags=["validate"],
)
app.include_router(
    preview.router,
    prefix=f"{settings.API_PREFIX}/preview",
    tags=["preview"],
)
app.include_router(
    glossaries.router,
    prefix=f"{settings.API_PREFIX}/glossaries",
    tags=["glossaries"],
)


@app.get("/metrics", tags=["health"])
async def metrics():
    """Prometheus metrics endpoint"""
    registry = CollectorRegistry()
    registry.register(REQUEST_COUNT)
    registry.register(REQUEST_DURATION)
    return Response(
        content=generate_latest(registry),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/", tags=["health"])
async def root():
    """Root endpoint"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "pipeline_version": settings.PIPELINE_VERSION,
        "environment": settings.APP_ENV,
    }


@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": request.url.path},
    )


@app.exception_handler(500)
async def server_error(request: Request, exc):
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "message": "An unexpected error occurred. Please try again later.",
        },
    )