import redis.asyncio as redis
from typing import Optional

from app.core.config import settings


class RedisClient:
    def __init__(self):
        self._client: Optional[redis.Redis] = None

    async def initialize(self):
        """Initialize Redis connection"""
        self._client = redis.from_url(
            settings.REDIS_URL,
            password=settings.REDIS_PASSWORD,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            decode_responses=True,
        )
        # Test connection
        await self._client.ping()

    async def get_client(self) -> redis.Redis:
        """Get Redis client instance"""
        if not self._client:
            await self.initialize()
        return self._client

    async def close(self):
        """Close Redis connection"""
        if self._client:
            await self._client.close()


redis_client = RedisClient()