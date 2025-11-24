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
        
        # Analyze Context
        # We override/enrich with provided query params if useful, but for now let's trust the analyzer
        # or pass them as hints? The current analyze_context doesn't take hints, but we could add them.
        # For now, we'll rely on the LLM analysis.
        
        manifest = await service.analyze_context(doc)
        
        # Override manifest if user provided specific context (optional enhancement)
        if topic != "General": manifest.topic = topic
        if industry != "General": manifest.industry = industry
        if country != "General": manifest.country = country

        # Correct
        # We process in chunks. For an API, we might want to stream or do it async.
        # For simplicity, we'll do it synchronously here (careful with timeouts).
        # A better approach for production is a job queue.
        
        # Chunking logic (simple 50 lines per chunk)
        chunk_size = 50
        corrected_segments = []
        
        for i in range(0, len(doc.segments), chunk_size):
            chunk = doc.segments[i:i+chunk_size]
            corrected_chunk = await service.correct_chunk(chunk, manifest)
            corrected_segments.extend(corrected_chunk)
            
        # Calculate Diffs
        changes = []
        original_segments = SubtitleParser.parse(content, fmt).segments # Re-parse to get original state
        
        # Map by index to be safe
        orig_map = {s.idx: s.text for s in original_segments}
        
        for corrected in corrected_segments:
            original_text = orig_map.get(corrected.idx, "")
            if original_text != corrected.text:
                # Simple word-based diff could be here, but for now let's return the full segment change
                # or try to find the specific changed phrase?
                # Let's do a simple check: if the change is small, return it.
                changes.append({
                    "id": corrected.idx,
                    "original": original_text,
                    "corrected": corrected.text
                })

        # Serialize
        output = ""
        for s in doc.segments:
            output += f"{s.idx}\n{s.start_ms} --> {s.end_ms}\n{s.text}\n\n" # Simplified SRT serialization
            
        # Cleanup
        os.unlink(tmp_path)
        
        return {
            "filename": file.filename,
            "context": manifest.dict(),
            "corrected_content": output,
            "changes": changes[:50] # Limit to top 50 changes to avoid payload bloat
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
