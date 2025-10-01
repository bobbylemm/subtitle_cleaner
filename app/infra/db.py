from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

engine = None
Async_SessionLocal = None


async def init_db():
    """Initialize database connection"""
    global engine, Async_SessionLocal
    engine = create_async_engine(
        settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
        echo=settings.DEBUG,
        pool_size=20,
        max_overflow=10,
    )
    Async_SessionLocal = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )


async def close_db():
    """Close database connection"""
    global engine
    if engine:
        await engine.dispose()


async def get_db() -> AsyncSession:
    """Dependency to get database session"""
    async with Async_SessionLocal() as session:
        yield session