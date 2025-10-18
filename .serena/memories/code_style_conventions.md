# Code Style and Conventions

## General Python Style
- **PEP 8 Compliance**: Follow Python PEP 8 style guidelines
- **Line Length**: 88 characters (Black formatter default)
- **Indentation**: 4 spaces, no tabs
- **Encoding**: UTF-8

## Code Formatting Tools
- **Black**: Automatic code formatting with 88-character line limit
- **isort**: Import statement sorting and organization
- **Ruff**: Fast Python linting for style and quality issues
- **mypy**: Static type checking

## Type Hints
- **Required**: All function parameters and return types must have type hints
- **Modern Syntax**: Use Python 3.10+ union syntax (`str | None` instead of `Optional[str]`)
- **Pydantic Models**: Use for API request/response schemas
- **SQLModel**: Use for database models with type safety

## Naming Conventions
- **Variables/Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private Members**: Leading underscore `_private_var`
- **Files/Modules**: `snake_case.py`
- **Directories**: `snake_case`

## Documentation
- **Docstrings**: Required for all public classes and functions
- **Format**: Google-style docstrings
- **Type Information**: Include parameter types and return types in docstrings
- **Examples**: Include usage examples for complex functions

Example:
```python
def process_subtitle_segment(
    segment: Segment, 
    settings: Settings
) -> Tuple[Segment, List[str]]:
    """Process a single subtitle segment with given settings.
    
    Args:
        segment: The subtitle segment to process
        settings: Processing configuration options
        
    Returns:
        Tuple of processed segment and list of applied modifications
        
    Raises:
        ValueError: If segment timing is invalid
    """
```

## Import Organization
- **Standard Library**: First group
- **Third-party**: Second group, separated by blank line
- **Local Imports**: Third group, separated by blank line
- **Relative Imports**: Use sparingly, prefer absolute imports

## Error Handling
- **Specific Exceptions**: Catch specific exception types, not broad `Exception`
- **HTTP Exceptions**: Use FastAPI's `HTTPException` for API errors
- **Logging**: Use structured logging with appropriate levels
- **Graceful Degradation**: ML features should degrade gracefully if models unavailable

## Async/Await Patterns
- **Async Functions**: Use for I/O bound operations (database, HTTP, file operations)
- **Database Sessions**: Always use async database sessions
- **Context Managers**: Use `async with` for resource management
- **Error Handling**: Proper exception handling in async contexts

## Configuration Management
- **Environment Variables**: Use Pydantic Settings for configuration
- **Validation**: Validate all configuration at startup
- **Defaults**: Provide sensible defaults for all settings
- **Documentation**: Document all environment variables in .env.sample

## Testing Conventions
- **Test Files**: `test_*.py` pattern
- **Test Classes**: `TestClassName` pattern
- **Test Methods**: `test_method_name` descriptive names
- **Fixtures**: Use pytest fixtures for common test setup
- **Async Testing**: Use `pytest-asyncio` for async test functions

## Database Conventions
- **Table Names**: `snake_case` (e.g., `subtitle_segments`)
- **Column Names**: `snake_case`
- **Foreign Keys**: Include table name (e.g., `user_id`)
- **Indexes**: Explicit naming convention
- **Migrations**: Descriptive names with timestamps

## API Design
- **REST Principles**: Follow RESTful design patterns
- **HTTP Status Codes**: Use appropriate status codes
- **Request/Response**: Use Pydantic models for validation
- **Error Responses**: Consistent error response format
- **Versioning**: Use URL path versioning (`/v1/`)

## Performance Considerations
- **Lazy Loading**: Load ML models only when needed
- **Caching**: Use Redis for frequently accessed data
- **Batching**: Process multiple segments together when possible
- **Memory Management**: Be mindful of memory usage with large files
- **Database Queries**: Use efficient queries with proper indexing