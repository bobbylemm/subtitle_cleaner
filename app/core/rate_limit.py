import asyncio
import time
from functools import wraps
from typing import Optional

from fastapi import HTTPException, Request, status
from fastapi.dependencies.utils import get_dependant, solve_dependencies

from app.core.config import settings
from app.infra.cache import redis_client


class RateLimiter:
    """
    Token bucket rate limiter with Redis backend.
    Implements per-key rate limiting with burst support.
    """

    def __init__(
        self,
        requests: Optional[int] = None,
        period: Optional[int] = None,
        burst: Optional[int] = None,
        concurrent: Optional[int] = None,
    ):
        self.requests = requests or settings.RATE_LIMIT_REQUESTS
        self.period = period or settings.RATE_LIMIT_PERIOD
        self.burst = burst or settings.RATE_LIMIT_BURST
        self.concurrent = concurrent or settings.RATE_LIMIT_CONCURRENT

    async def check_rate_limit(self, key: str) -> tuple[bool, dict]:
        """
        Check if request is allowed under rate limit.
        Returns (allowed, metadata) tuple.
        """
        if not settings.RATE_LIMIT_ENABLED:
            return True, {"limit": -1, "remaining": -1, "reset": 0}

        redis = await redis_client.get_client()
        now = time.time()

        # Keys for token bucket and concurrent requests
        bucket_key = f"rate_limit:bucket:{key}"
        concurrent_key = f"rate_limit:concurrent:{key}"
        last_refill_key = f"rate_limit:last_refill:{key}"

        async with redis.pipeline(transaction=True) as pipe:
            # Get current state
            await pipe.get(bucket_key)
            await pipe.get(last_refill_key)
            await pipe.get(concurrent_key)
            results = await pipe.execute()

            current_tokens = float(results[0]) if results[0] else float(self.burst)
            last_refill = float(results[1]) if results[1] else now
            concurrent_count = int(results[2]) if results[2] else 0

            # Check concurrent limit
            if concurrent_count >= self.concurrent:
                return False, {
                    "limit": self.requests,
                    "remaining": 0,
                    "reset": int(now + self.period),
                    "retry_after": self.period,
                    "concurrent_limit_exceeded": True,
                }

            # Calculate tokens to add based on time elapsed
            time_elapsed = now - last_refill
            tokens_to_add = (time_elapsed / self.period) * self.requests
            new_tokens = min(current_tokens + tokens_to_add, self.burst)

            # Check if we have enough tokens
            if new_tokens < 1:
                reset_time = last_refill + self.period
                return False, {
                    "limit": self.requests,
                    "remaining": 0,
                    "reset": int(reset_time),
                    "retry_after": int(reset_time - now),
                }

            # Consume a token
            new_tokens -= 1

            # Update bucket state
            pipe.multi()
            await pipe.setex(bucket_key, self.period * 2, new_tokens)
            await pipe.setex(last_refill_key, self.period * 2, now)
            await pipe.execute()

            return True, {
                "limit": self.requests,
                "remaining": int(new_tokens),
                "reset": int(now + self.period),
            }

    async def acquire_concurrent_slot(self, key: str) -> bool:
        """Acquire a concurrent processing slot"""
        if not settings.RATE_LIMIT_ENABLED:
            return True

        redis = await redis_client.get_client()
        concurrent_key = f"rate_limit:concurrent:{key}"

        current = await redis.incr(concurrent_key)
        await redis.expire(concurrent_key, self.period)

        if current > self.concurrent:
            await redis.decr(concurrent_key)
            return False

        return True

    async def release_concurrent_slot(self, key: str):
        """Release a concurrent processing slot"""
        if not settings.RATE_LIMIT_ENABLED:
            return

        redis = await redis_client.get_client()
        concurrent_key = f"rate_limit:concurrent:{key}"
        
        current = await redis.decr(concurrent_key)
        if current < 0:
            await redis.delete(concurrent_key)


def rate_limit(
    requests: Optional[int] = None,
    period: Optional[int] = None,
    burst: Optional[int] = None,
    concurrent: Optional[int] = None,
    key_func: Optional[callable] = None,
):
    """
    Decorator for rate limiting endpoints.
    
    Args:
        requests: Number of requests allowed per period
        period: Time period in seconds
        burst: Maximum burst size
        concurrent: Maximum concurrent requests
        key_func: Function to extract rate limit key from request
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request object
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request:
                request = kwargs.get("request")

            if not request:
                # No request object, skip rate limiting
                return await func(*args, **kwargs)

            # Determine rate limit key
            if key_func:
                limit_key = key_func(request)
            else:
                # Default: use client IP or API key
                from app.core.security import optional_api_key

                api_key = await optional_api_key(request)
                if api_key:
                    limit_key = f"api_key:{api_key}"
                else:
                    client_ip = request.client.host if request.client else "unknown"
                    limit_key = f"ip:{client_ip}"

            # Apply rate limit
            limiter = RateLimiter(requests, period, burst, concurrent)
            allowed, metadata = await limiter.check_rate_limit(limit_key)

            # Add rate limit headers to response
            request.state.rate_limit_headers = {
                "X-RateLimit-Limit": str(metadata["limit"]),
                "X-RateLimit-Remaining": str(metadata["remaining"]),
                "X-RateLimit-Reset": str(metadata["reset"]),
            }

            if not allowed:
                retry_after = metadata.get("retry_after", 60)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(metadata["limit"]),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(metadata["reset"]),
                    },
                )

            # For concurrent limits, acquire and release slot
            if concurrent:
                acquired = await limiter.acquire_concurrent_slot(limit_key)
                if not acquired:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Concurrent request limit exceeded",
                        headers={"Retry-After": str(period or settings.RATE_LIMIT_PERIOD)},
                    )

                try:
                    result = await func(*args, **kwargs)
                finally:
                    await limiter.release_concurrent_slot(limit_key)
                return result

            return await func(*args, **kwargs)

        return wrapper

    return decorator


class RateLimitDependency:
    """
    FastAPI dependency for rate limiting.
    Can be used as an alternative to the decorator.
    """

    def __init__(
        self,
        requests: Optional[int] = None,
        period: Optional[int] = None,
        burst: Optional[int] = None,
        concurrent: Optional[int] = None,
    ):
        self.limiter = RateLimiter(requests, period, burst, concurrent)

    async def __call__(self, request: Request, api_key: Optional[str] = None):
        """Check rate limit as a dependency"""
        # Determine rate limit key
        if api_key:
            limit_key = f"api_key:{api_key}"
        else:
            client_ip = request.client.host if request.client else "unknown"
            limit_key = f"ip:{client_ip}"

        allowed, metadata = await self.limiter.check_rate_limit(limit_key)

        # Store headers in request state for middleware to add
        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(metadata["limit"]),
            "X-RateLimit-Remaining": str(metadata["remaining"]),
            "X-RateLimit-Reset": str(metadata["reset"]),
        }

        if not allowed:
            retry_after = metadata.get("retry_after", 60)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(metadata["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(metadata["reset"]),
                },
            )

        return limit_key


# Pre-configured rate limiters for common use cases
standard_rate_limit = RateLimitDependency()
strict_rate_limit = RateLimitDependency(requests=10, period=60, burst=5)
preview_rate_limit = RateLimitDependency(requests=50, period=60, burst=10)