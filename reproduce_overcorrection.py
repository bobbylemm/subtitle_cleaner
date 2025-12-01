import asyncio
import os
from app.services.universal_corrector import UniversalCorrectionService
from app.domain.constants import SubtitleFormat
from app.core.config import settings

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

async def verify():
    if not settings.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set in settings")
        return

    service = UniversalCorrectionService()
    
    with open("strictness_sample.srt", "r") as f:
        content = f.read()
        
    print("\n=== Processing Strictness Sample ===")
    result = await service.process_full_document(content, SubtitleFormat.SRT)
    
    print(f"\nDetected Dialect: {result.manifest.dialect}")
    
    print("\n--- Corrections ---")
    for change in result.changes:
        print(f"[{change['id']}] '{change['original']}' -> '{change['corrected']}'")
        
    # Validation Logic
    content_lower = result.content.lower()
    
    # Check "Manchester City"
    if "manchester city" in content_lower and "man city" not in content_lower:
        print("SUCCESS: 'Manchester City' preserved (not shortened).")
    elif "man city" in content_lower:
        print("FAILURE: 'Manchester City' shortened to 'Man City'.")
    else:
        print("WARNING: 'Manchester City' not found??")

    # Check "Guardiola"
    if "guardiola" in content_lower:
         # Check if it was changed to something else in the changes list
         guardiola_changed = False
         for change in result.changes:
             if "guardiola" in change['original'].lower() and "guardiola" not in change['corrected'].lower():
                 guardiola_changed = True
                 print(f"FAILURE: 'Guardiola' changed to '{change['corrected']}'")
         
         if not guardiola_changed:
             print("SUCCESS: 'Guardiola' preserved.")
    else:
        print("FAILURE: 'Guardiola' removed or corrupted.")

if __name__ == "__main__":
    asyncio.run(verify())
