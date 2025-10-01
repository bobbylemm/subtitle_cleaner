# PostgreSQL Setup for Tenant Memory

## Overview

The tenant memory system (Layer 6) now supports PostgreSQL for production-grade persistence. This replaces the file-based storage with a scalable database solution.

## Database Schema

### Tables Created

1. **tenant_corrections**: Stores learned corrections per tenant
   - `tenant_id`: Identifies the tenant
   - `original_text`: The incorrect text
   - `corrected_text`: The correct replacement
   - `confidence`: Float score (0.0-1.0)
   - `confidence_level`: Enum (LOW, MEDIUM, HIGH, VERY_HIGH)
   - `usage_count`: Number of times used
   - `source`: Where the correction came from
   - `context`: Optional context string

2. **tenant_context**: Stores tenant-specific glossary/context
   - `tenant_id`: Identifies the tenant
   - `entity_text`: The entity/term
   - `entity_type`: Optional classification
   - `authority_score`: Confidence in this context

3. **correction_history**: Tracks correction performance
   - `tenant_id`: Identifies the tenant
   - `original_text`, `corrected_text`: What was corrected
   - `was_successful`: Whether it worked
   - `applied_at`: Timestamp

## Running Migrations

### Step 1: Start Docker Services

```bash
docker compose up -d
```

### Step 2: Run Database Migrations

```bash
# Run migrations inside the API container
docker compose exec api alembic upgrade head
```

### Step 3: Verify Migration Status

```bash
# Check current migration version
docker compose exec api alembic current

# View migration history
docker compose exec api alembic history
```

## Rolling Back Migrations

```bash
# Rollback one migration
docker compose exec api alembic downgrade -1

# Rollback to specific version
docker compose exec api alembic downgrade <revision_id>

# Rollback all migrations
docker compose exec api alembic downgrade base
```

## Creating New Migrations

```bash
# Auto-generate migration from model changes
docker compose exec api alembic revision --autogenerate -m "description"

# Create empty migration
docker compose exec api alembic revision -m "description"
```

## Configuration

The system automatically uses PostgreSQL when:
- `ENABLE_TENANT_MEMORY=true` (in .env)
- Database connection is available
- `use_postgresql=True` in EnhancedCleaningConfig (default)

### Environment Variables

```bash
DATABASE_URL=postgresql://postgres:postgres@db:5432/subtitle_cleaner
ENABLE_TENANT_MEMORY=true
```

## Fallback to File Storage

If PostgreSQL is unavailable, the system automatically falls back to file-based storage in `./tenant_data/`.

## Testing

```bash
# Test with tenant memory enabled
curl -X POST http://localhost:8080/v1/clean/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-dev-key-1234567890" \
  -d '{
    "content": "1\n00:00:00,000 --> 00:00:03,000\nOpus Odima is the governor",
    "format": "srt",
    "language": "en",
    "tenant_id": "test_tenant",
    "context_sources": [{
      "type": "TEXT",
      "content": "Hope Uzodinma is the governor of Imo State.",
      "source_id": "ctx1",
      "authority": 0.9
    }]
  }'
```

## Monitoring

### Check Database Connectivity

```bash
docker compose exec api python -c "
from app.infra.db import init_db
import asyncio
asyncio.run(init_db())
print('Database connection successful!')
"
```

### View Tenant Corrections

```bash
docker compose exec db psql -U postgres -d subtitle_cleaner -c "
SELECT tenant_id, original_text, corrected_text, confidence_level, usage_count 
FROM tenant_corrections 
ORDER BY created_at DESC 
LIMIT 10;
"
```

### Check Migration Status

```bash
docker compose exec db psql -U postgres -d subtitle_cleaner -c "
SELECT version_num, applied_on FROM alembic_version;
"
```

## Architecture

### Dual Backend Support

The system supports two backends:

1. **PostgreSQL** (Production): `TenantMemoryPostgreSQL`
   - Scalable for multiple instances
   - ACID transactions
   - Better query performance

2. **File Storage** (Development): `TenantMemory`
   - Simple file-based JSON storage
   - No database required
   - Good for local development

### Backend Selection Logic

```python
if config.enable_tenant_memory:
    if config.use_postgresql and db_session:
        # Use PostgreSQL
        tenant_memory_pg = TenantMemoryPostgreSQL()
    else:
        # Fallback to file storage
        tenant_memory = TenantMemory(storage_path="./tenant_data")
```

## Troubleshooting

### Migration Fails

```bash
# Check database logs
docker compose logs db

# Verify database is ready
docker compose exec db pg_isready -U postgres

# Check alembic version table
docker compose exec db psql -U postgres -d subtitle_cleaner -c "
SELECT * FROM alembic_version;
"
```

### Connection Issues

```bash
# Test database connection
docker compose exec api python -c "
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings

async def test():
    engine = create_async_engine(settings.DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://'))
    async with engine.connect() as conn:
        result = await conn.execute('SELECT 1')
        print('Connection successful!')
    await engine.dispose()

asyncio.run(test())
"
```

## Performance Considerations

- **Indexes**: Created on `tenant_id`, `original_text`, `confidence` for fast queries
- **Connection Pooling**: Configured with 20 connections, 10 max overflow
- **Async Operations**: All database operations are async for better concurrency

## Production Recommendations

1. **Backup Strategy**: Regular PostgreSQL backups
2. **Monitoring**: Track query performance and connection pool usage
3. **Scaling**: Use connection pooling and read replicas for high load
4. **Security**: Use SSL/TLS for database connections in production