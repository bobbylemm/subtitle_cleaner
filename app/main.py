import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.api.routers import health, validate, universal
from app.core.config import settings

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
    setup_metrics()

    yield
    
    # Shutdown
    
    
# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


app = FastAPI(
    title=settings.APP_NAME,
    description="Universal Subtitle Corrector API",
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "universal", "description": "Universal Subtitle Correction"},
        {"name": "validate", "description": "Validate subtitle files"},
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
    universal.router,
    prefix=f"{settings.API_PREFIX}/universal",
    tags=["universal"],
)
app.include_router(
    validate.router,
    prefix=f"{settings.API_PREFIX}/validate",
    tags=["validate"],
)

# Mount Static Files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("app/static/index.html")


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





@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": request.url.path},
    )


@app.exception_handler(500)
async def server_error(request: Request, exc):
    request_id = getattr(request.state, "request_id", "unknown")
    import logging
    logging.error(f"Unhandled exception (Request ID: {request_id}): {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "message": "An unexpected error occurred. Please try again later.",
        },
    )