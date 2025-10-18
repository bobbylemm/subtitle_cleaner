# API Endpoints

## Base Configuration
- **Base URL**: `http://localhost:8080`
- **API Prefix**: `/v1`
- **Authentication**: API Key via `X-API-Key` header
- **Documentation**: `/docs` (Swagger UI) and `/redoc` (ReDoc) - available in debug mode only

## Endpoint Categories

### Health & Monitoring
- `GET /` - Root endpoint with service information
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics endpoint

### Core Processing
- `POST /v1/clean/` - **Main cleaning endpoint**
  - Input: Subtitle content (SRT/WebVTT), language, processing settings
  - Output: Cleaned content with processing report and statistics
  - Features: Enhanced ML processing, context sources, tenant memory

### Validation
- `POST /v1/validate/` - Validate subtitle files
  - Input: Subtitle content and validation settings
  - Output: Validation issues, statistics, and recommendations

### Preview
- `POST /v1/preview/` - Preview cleaning results without applying changes
  - Input: Same as clean endpoint
  - Output: Side-by-side comparison of original vs cleaned content

### Glossary Management
- `GET /v1/glossaries/` - List available glossaries
- `POST /v1/glossaries/` - Create new glossary
- `GET /v1/glossaries/{id}` - Get specific glossary
- `PUT /v1/glossaries/{id}` - Update glossary
- `DELETE /v1/glossaries/{id}` - Delete glossary
- `POST /v1/glossaries/{id}/apply` - Apply glossary to content

## Request Features

### Enhanced Processing Options
- **Context Sources**: Provide URLs, files, or text for entity extraction
- **Retrieval Mode**: Enable Wikipedia/Wikidata lookup for suspicious entities
- **LLM Selection**: Use OpenAI for ambiguous entity disambiguation
- **Tenant Memory**: Per-tenant learning and customization
- **Context Mode**: auto, manual, hybrid, smart, none

### Processing Settings
- **Language Support**: en, es, fr, de, it, pt, nl
- **Merge Modes**: smart, aggressive, conservative, off
- **Quality Controls**: CPS limits, duration constraints, line wrapping
- **ML Features**: Punctuation, grammar correction, filler detection