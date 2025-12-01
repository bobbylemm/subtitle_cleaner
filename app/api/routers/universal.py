from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import Optional
import tempfile
import os
import shutil
from app.services.universal_corrector import UniversalCorrectionService
from app.services.parser import SubtitleParser
from app.domain.constants import SubtitleFormat

router = APIRouter()

@router.post("/universal-correct", summary="Correct subtitles using Universal LLM Corrector")
async def universal_correct(
    file: UploadFile = File(...),
    topic: str = "General",
    industry: str = "General",
    country: str = "General",
    background_tasks: BackgroundTasks = None
):
    """
    Upload a subtitle file and get a corrected version using the Universal Correction Service.
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Parse
        with open(tmp_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Detect format
        fmt = SubtitleFormat.SRT if tmp_path.endswith(".srt") else SubtitleFormat.WEBVTT
        doc = SubtitleParser.parse(content, fmt)

        # Initialize Service
        service = UniversalCorrectionService()
        
        # Process
        result = await service.process_full_document(content, fmt)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return {
            "filename": file.filename,
            "context": result.manifest.dict(),
            "corrected_content": result.content,
            "changes": result.changes[:50], # Limit to top 50 changes
            "applied_corrections": [c.dict() for c in result.applied_corrections]
        }

    except Exception as e:
        import logging
        logging.error(f"Universal Correction Failed: {e}", exc_info=True)
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))
