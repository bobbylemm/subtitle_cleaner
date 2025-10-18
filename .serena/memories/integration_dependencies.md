# Integration Points and Dependencies

## External Service Dependencies

### Required Services
- **PostgreSQL 16**: Primary database for tenant memory, glossaries, and persistent storage
- **Redis 7**: Caching, rate limiting, and session management
- **Docker**: Container orchestration for multi-service deployment

### Optional External APIs
- **OpenAI API** (Layer 5): 
  - Required for LLM-based entity disambiguation
  - Configurable via `OPENAI_API_KEY` environment variable
  - Falls back to rule-based processing if unavailable
  - Used selectively to minimize costs

- **Wikipedia/Wikidata** (Layer 4):
  - On-demand retrieval for suspicious entity verification
  - Requires internet connectivity
  - Falls back gracefully if unavailable
  - Implements caching to reduce API calls

## Internal Integration Points

### Database Integration
- **SQLModel/SQLAlchemy**: Type-safe ORM with async support
- **Alembic**: Database migration management
- **Connection Pooling**: Managed via asyncpg driver
- **Health Checks**: Database connectivity monitoring

### Caching Layer
- **Redis Integration**: 
  - Cache keys follow hierarchical naming (e.g., `tenant:{id}:corrections`)
  - TTL-based expiration (15 minutes default)
  - Fallback to database when cache misses
  - Rate limiting state storage

### ML Model Integration
- **Model Loading**: Lazy loading on first use to reduce startup time
- **ONNX Runtime**: Optimized inference for production models
- **Hugging Face Transformers**: NLP model integration
- **GPU Support**: Optional CUDA acceleration for high-volume processing

## API Integration Patterns

### Authentication & Authorization
- **API Key Header**: `X-API-Key` header for all protected endpoints
- **Rate Limiting**: Redis-backed sliding window implementation
- **Request ID Tracking**: Unique request IDs for tracing and debugging

### Request/Response Flow
1. **Authentication**: API key validation
2. **Rate Limiting**: Check request limits
3. **Input Validation**: Pydantic schema validation
4. **Processing Pipeline**: Layer 1-6 processing chain
5. **Response Generation**: Structured response with metadata

### Error Handling Integration
- **Graceful Degradation**: ML features fall back to rule-based processing
- **Circuit Breaker**: Automatic service degradation on repeated failures
- **Timeout Management**: Processing timeouts with partial results
- **Error Reporting**: Structured error responses with request IDs

## Data Flow Integration

### Context Sources (Layer 3)
- **URL Extraction**: Web scraping with BeautifulSoup
- **File Processing**: Support for various document formats
- **Text Processing**: Direct text input processing
- **Entity Extraction**: spaCy NLP pipeline integration

### Retrieval Engine (Layer 4)
- **Wikipedia API**: Structured data retrieval
- **Wikidata SPARQL**: Entity resolution queries
- **Confidence Scoring**: Multi-source consensus building
- **Regional Hints**: Location-aware entity disambiguation

### Tenant Memory (Layer 6)
- **PostgreSQL Storage**: Persistent correction storage
- **Learning Algorithm**: Confidence-based learning from corrections
- **Cross-Session Persistence**: Tenant preferences across API calls
- **Conflict Resolution**: Handling conflicting correction suggestions

## Monitoring & Observability

### Metrics Integration
- **Prometheus Metrics**: Request counters, latency histograms, error rates
- **Custom Metrics**: Processing time, segment counts, ML model usage
- **Health Endpoints**: Service and dependency health monitoring

### Logging Integration
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Log Levels**: Configurable logging verbosity
- **Error Tracking**: Exception capture with stack traces
- **Performance Logging**: Processing pipeline timing

### Tracing Integration
- **OpenTelemetry**: Distributed tracing across components
- **Request Correlation**: End-to-end request tracking
- **Database Query Tracing**: SQL query performance monitoring
- **ML Model Tracing**: Model inference timing and results

## Security Integration

### Input Validation
- **File Size Limits**: Configurable maximum file sizes (10MB default)
- **Content Validation**: Malicious content detection
- **Encoding Validation**: UTF-8 encoding enforcement
- **Rate Limiting**: Request rate and concurrent connection limits

### Data Protection
- **No Data Persistence**: Subtitle content not stored by default
- **Tenant Isolation**: Per-tenant data segregation
- **API Key Security**: Secure key validation and storage
- **CORS Configuration**: Configurable cross-origin policies

## Deployment Integration

### Container Orchestration
- **Docker Compose**: Multi-service development setup
- **Health Checks**: Container health monitoring
- **Volume Management**: Persistent storage for database and logs
- **Network Isolation**: Service-to-service communication

### Environment Configuration
- **12-Factor App**: Environment-based configuration
- **Secret Management**: Secure handling of API keys and credentials
- **Feature Flags**: Environment-based feature enabling/disabling
- **Resource Limits**: Configurable memory and CPU constraints