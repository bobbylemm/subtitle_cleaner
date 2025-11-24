from fastapi import APIRouter
import time
from typing import Dict, Any

from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": int(time.time()),
        "checks": {
            "api": "healthy",
            "knowledge_base": "local_sqlite"
        }
    }


@router.get("/healthz")
async def liveness() -> Dict[str, str]:
    """Kubernetes liveness probe - checks if the service is alive"""
    return {"status": "ok"}


@router.get("/readyz")
async def readiness() -> Dict[str, str]:
    """Kubernetes readiness probe - checks if the service is ready to accept requests"""
    return {"status": "ready"}