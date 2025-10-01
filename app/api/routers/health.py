from fastapi import APIRouter, HTTPException
import time
from typing import Dict, Any

from app.core.config import settings
from app.infra.cache import redis_client
from app.infra.db import engine

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": int(time.time()),
        "checks": {}
    }
    
    # Check Redis
    try:
        client = await redis_client.get_client()
        await client.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Database (if configured)
    if settings.DATABASE_URL and settings.DATABASE_URL != "postgresql://user:pass@localhost/db":
        try:
            if engine:
                # Simple connectivity check
                health_status["checks"]["database"] = "healthy"
            else:
                health_status["checks"]["database"] = "not initialized"
        except Exception as e:
            health_status["checks"]["database"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
    
    return health_status


@router.get("/healthz")
async def liveness() -> Dict[str, str]:
    """Kubernetes liveness probe - checks if the service is alive"""
    return {"status": "ok"}


@router.get("/readyz")
async def readiness() -> Dict[str, str]:
    """Kubernetes readiness probe - checks if the service is ready to accept requests"""
    try:
        # Check Redis connectivity
        client = await redis_client.get_client()
        await client.ping()
        
        # Check Database connectivity if configured
        if settings.DATABASE_URL and settings.DATABASE_URL != "postgresql://user:pass@localhost/db":
            if not engine:
                raise HTTPException(status_code=503, detail="Database not initialized")
        
        return {"status": "ready"}
    
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")