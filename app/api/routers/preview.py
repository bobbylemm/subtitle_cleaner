from fastapi import APIRouter, Depends, HTTPException
import time

from app.core.security import require_api_key
from app.core.rate_limit import preview_rate_limit
from app.api.schemas.subtitles import PreviewRequest, PreviewResponse, Segment as APISegment
from app.domain.models import Settings
from app.services.parser import SubtitleParser
from app.services.cleaner import SubtitleCleaner
from app.infra.metrics import PROCESSING_TIME

router = APIRouter()


@router.post(
    "/",
    response_model=PreviewResponse,
    dependencies=[Depends(require_api_key), Depends(preview_rate_limit)]
)
async def preview_cleaning(request: PreviewRequest) -> PreviewResponse:
    """Preview subtitle cleaning without full processing"""
    start_time = time.time()
    
    try:
        # Parse input
        try:
            document = SubtitleParser.parse(request.content, request.format)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse subtitles: {str(e)}")
        
        if not document.segments:
            raise HTTPException(status_code=400, detail="No segments found in input")
        
        # Select segments for preview
        if request.segment_indices:
            # Filter to requested segments
            preview_segments = [
                seg for seg in document.segments 
                if seg.idx in request.segment_indices
            ][:request.max_segments]
        else:
            # Take first N segments
            preview_segments = document.segments[:request.max_segments]
        
        if not preview_segments:
            raise HTTPException(status_code=400, detail="No segments to preview")
        
        # Convert to API format for original
        original = [
            APISegment(
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                text=seg.text
            )
            for seg in preview_segments
        ]
        
        # Create settings and clean
        settings = Settings(
            language=request.language.value,
            **request.settings if request.settings else {}
        )
        
        cleaner = SubtitleCleaner(settings)
        
        # Create a temporary document with just preview segments
        from app.domain.models import SubtitleDocument
        preview_doc = SubtitleDocument(segments=preview_segments)
        
        # Clean the preview
        cleaned_doc = cleaner.clean(preview_doc)
        
        # Convert cleaned segments to API format
        cleaned = [
            APISegment(
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                text=seg.text
            )
            for seg in cleaned_doc.segments
        ]
        
        # Calculate changes
        changes = []
        for i, (orig, clean) in enumerate(zip(preview_segments, cleaned_doc.segments)):
            if orig.text != clean.text or orig.end_ms != clean.end_ms:
                changes.append({
                    "segment_index": orig.idx,
                    "type": "modified",
                    "original_text": orig.text,
                    "cleaned_text": clean.text,
                    "timing_changed": orig.end_ms != clean.end_ms or orig.start_ms != clean.start_ms
                })
        
        # Calculate reduction
        original_chars = sum(len(seg.text) for seg in preview_segments)
        cleaned_chars = sum(len(seg.text) for seg in cleaned_doc.segments)
        reduction_percent = ((original_chars - cleaned_chars) / original_chars * 100) if original_chars > 0 else 0
        
        # Record processing time
        processing_time = time.time() - start_time
        PROCESSING_TIME.labels(
            operation="preview",
            language=request.language.value
        ).observe(processing_time)
        
        return PreviewResponse(
            original=original,
            cleaned=cleaned,
            changes=changes,
            estimated_reduction_percent=max(0, reduction_percent)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")