from fastapi import APIRouter, Depends, HTTPException
import time

from app.core.security import require_api_key
from app.core.rate_limit import standard_rate_limit
from app.api.schemas.subtitles import ValidateRequest, ValidateResponse
from app.domain.models import Settings
from app.services.parser import SubtitleParser
from app.services.validator import SubtitleValidator
from app.infra.metrics import PROCESSING_TIME, FILE_SIZE

router = APIRouter()


@router.post(
    "/",
    response_model=ValidateResponse,
    dependencies=[Depends(require_api_key), Depends(standard_rate_limit)]
)
async def validate_subtitles(request: ValidateRequest) -> ValidateResponse:
    """Validate subtitle content for errors and issues"""
    start_time = time.time()
    
    try:
        # Parse input
        try:
            document = SubtitleParser.parse(request.content, request.format)
        except Exception as e:
            # Parse error is a validation error
            return ValidateResponse(
                valid=False,
                format=request.format,
                language=request.language,
                errors=[{
                    "segment_index": None,
                    "issue_type": "parse_error",
                    "severity": "error",
                    "message": f"Failed to parse subtitles: {str(e)}",
                    "details": {}
                }],
                warnings=[],
                stats={}
            )
        
        # Record metrics
        FILE_SIZE.labels(format=request.format.value).observe(len(request.content))
        
        # Create settings
        settings = Settings(language=request.language.value)
        
        # Validate
        validator = SubtitleValidator(settings)
        issues = validator.validate(document, strict=request.strict)
        
        # Separate errors and warnings
        errors = []
        warnings = []
        
        for issue in issues:
            issue_dict = issue.to_dict()
            if issue.severity == "error":
                errors.append(issue_dict)
            else:
                warnings.append(issue_dict)
        
        # Get statistics
        stats = validator.get_stats(document)
        
        # Record processing time
        processing_time = time.time() - start_time
        PROCESSING_TIME.labels(
            operation="validate",
            language=request.language.value
        ).observe(processing_time)
        
        stats["processing_time_ms"] = int(processing_time * 1000)
        
        return ValidateResponse(
            valid=len(errors) == 0,
            format=request.format,
            language=request.language,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")