from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional, List
import time
import traceback

from app.core.security import require_api_key
from app.core.rate_limit import standard_rate_limit
from app.core.config import settings as app_settings
from app.infra.db import get_db
from app.infra.metrics import PROCESSING_TIME, FILE_SIZE, SEGMENT_COUNT, ERROR_COUNT
from app.api.schemas.subtitles import CleanRequest, CleanResponse
from app.api.schemas.settings import ProcessingSettings
from app.domain.models import Settings
from app.domain.constants import SubtitleFormat, Language
from app.services.parser import SubtitleParser, SubtitleSerializer
from app.services.validator import SubtitleValidator
from app.services.cleaner import SubtitleCleaner
from app.services.glossary import glossary_enforcer
from app.services.enhanced_cleaner import EnhancedSubtitleCleaner, EnhancedCleaningConfig
from app.services.context_extraction_improved import (
    ContextSource, 
    SourceType
)
from app.services.tenant_memory import ConfidenceLevel
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.post(
    "/",
    response_model=CleanResponse,
    dependencies=[Depends(require_api_key), Depends(standard_rate_limit)]
)
async def clean_subtitles(
    request: CleanRequest,
    db: AsyncSession = Depends(get_db)
) -> CleanResponse:
    """Clean and process subtitle content"""
    start_time = time.time()
    
    try:
        # Decode content if base64 encoded
        content = request.content
        if request.is_base64:
            import base64
            try:
                content = base64.b64decode(content).decode('utf-8')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode base64 content: {str(e)}")
        
        # Parse input
        try:
            document = SubtitleParser.parse(content, request.format)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse subtitles: {str(e)}")
        
        if not document.segments:
            raise HTTPException(status_code=400, detail="No segments found in input")
        
        # Record metrics
        FILE_SIZE.labels(format=request.format.value).observe(len(request.content))
        SEGMENT_COUNT.labels(operation="clean").observe(len(document.segments))
        
        # Convert API settings to domain settings
        settings = Settings(
            language=request.language.value,
            **request.settings if request.settings else {}
        )
        
        # Validate
        validator = SubtitleValidator(settings)
        validation_issues = validator.validate(document)
        
        # Check if enhanced features are requested
        # Debug with print statements
        print(f"DEBUG: Request type: {type(request)}")
        print(f"DEBUG: Request has context_sources attr: {hasattr(request, 'context_sources')}")
        print(f"DEBUG: Context sources value: {getattr(request, 'context_sources', 'NOT FOUND')}")
        
        # Check each field explicitly
        has_context = hasattr(request, 'context_sources') and request.context_sources
        has_retrieval = hasattr(request, 'enable_retrieval') and request.enable_retrieval
        has_llm = hasattr(request, 'enable_llm') and request.enable_llm  
        has_tenant = hasattr(request, 'tenant_id') and request.tenant_id
        
        print(f"DEBUG: has_context={has_context}, has_retrieval={has_retrieval}, has_llm={has_llm}, has_tenant={has_tenant}")
        
        use_enhanced = any([has_context, has_retrieval, has_llm, has_tenant])
        print(f"DEBUG: Enhanced features will be used: {use_enhanced}")
        
        # Always apply basic cleaning first (Layers 1-2)
        cleaner = SubtitleCleaner(settings)
        cleaned_document = await cleaner.clean(document)
        modifications = cleaner.get_modifications_report()
        
        if use_enhanced:
            # Apply enhanced features (Layers 3-6) on top of basic cleaning
            config = EnhancedCleaningConfig()
            
            # Configure context extraction (Layer 3)
            if hasattr(request, 'context_sources') and request.context_sources:
                config.enable_context_extraction = True
                config.context_sources = [
                    ContextSource(
                        source_type=SourceType[source.get('source_type', source.get('type', 'TEXT')).upper()],
                        content=source.get('content', ''),
                        source_id=source.get('source_id', f"source_{i}"),
                        authority_score=source.get('authority_score', source.get('authority', 1.0))
                    )
                    for i, source in enumerate(request.context_sources)
                ]
            
            # Configure retrieval (Layer 4) - use settings from environment
            config.enable_retrieval = getattr(request, 'enable_retrieval', app_settings.ENABLE_RETRIEVAL)
            config.region_hint = getattr(request, 'region_hint', None)
            config.max_retrieval_spans = app_settings.RETRIEVAL_MAX_SPANS
            
            # Configure LLM selection (Layer 5) - use settings from environment
            config.enable_llm_selection = getattr(request, 'enable_llm', app_settings.ENABLE_LLM_SELECTION)
            config.openai_api_key = getattr(request, 'openai_api_key', app_settings.OPENAI_API_KEY)
            config.llm_model = app_settings.OPENAI_MODEL
            
            # Configure tenant memory (Layer 6) - use settings from environment
            config.enable_tenant_memory = app_settings.ENABLE_TENANT_MEMORY
            tenant_id = getattr(request, 'tenant_id', None)
            if tenant_id:
                config.tenant_id = tenant_id
            
            # Create enhanced cleaner with database session
            enhanced_cleaner = EnhancedSubtitleCleaner(config, db_session=db)
            
            # Convert segments to dict format for enhanced processing
            segments_dict = [
                {
                    'idx': i,
                    'start': seg.start_ms / 1000.0,  # Convert ms to seconds
                    'end': seg.end_ms / 1000.0,      # Convert ms to seconds
                    'text': seg.text
                }
                for i, seg in enumerate(cleaned_document.segments)
            ]
            
            # Apply enhanced features
            enhanced_segments, enhanced_metadata = await enhanced_cleaner.clean_subtitles(
                segments_dict,
                language=request.language.value,
                tenant_id=tenant_id
            )
            
            # Update document with enhanced corrections
            for i, seg in enumerate(cleaned_document.segments):
                seg.text = enhanced_segments[i]['text']
            
            # Merge metadata
            modifications.update(enhanced_metadata)
        
        # Apply glossary if specified
        if settings.glossary:
            for segment in cleaned_document.segments:
                segment.text, _ = glossary_enforcer.apply_glossary(
                    segment.text,
                    settings.glossary
                )
        
        # Serialize output
        output_content = SubtitleSerializer.serialize(cleaned_document, request.format)
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Record processing metrics
        PROCESSING_TIME.labels(
            operation="clean",
            language=request.language.value
        ).observe(processing_time_ms / 1000)
        
        # Prepare response
        segments_modified = modifications.get("summary", {}).get("total_modifications", 0)
        if use_enhanced:
            segments_modified = len(modifications.get("corrections_made", {}))
        
        return CleanResponse(
            success=True,
            content=output_content,
            format=request.format,
            segments_processed=len(document.segments),
            segments_modified=segments_modified,
            processing_time_ms=processing_time_ms,
            report={
                "modifications": modifications,
                "validation_issues": [issue.to_dict() for issue in validation_issues],
                "stats": validator.get_stats(cleaned_document),
                "enhanced_features": modifications if use_enhanced else None
            },
            errors=[]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # Log error
        ERROR_COUNT.labels(
            error_type="processing_error",
            operation="clean"
        ).inc()
        
        # Return error response
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return CleanResponse(
            success=False,
            content=None,
            format=request.format,
            segments_processed=0,
            segments_modified=0,
            processing_time_ms=processing_time_ms,
            report=None,
            errors=[str(e), traceback.format_exc()]
        )