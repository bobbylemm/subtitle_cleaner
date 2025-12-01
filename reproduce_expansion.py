import asyncio
import logging
from app.services.universal_corrector import UniversalCorrectionService
from app.domain.constants import SubtitleFormat
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)

async def verify():
    if not settings.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set in settings")
        return

    service = UniversalCorrectionService()
    
    # Sample text where "Man City" is used and should be preserved
    content = """1
00:00:00,000 --> 00:00:02,000
I think Man City played well today.
"""
        
    print("\n=== Processing Expansion Sample ===")
    result = await service.process_full_document(content, SubtitleFormat.SRT)
    
    print(f"\nDetected Style: {result.manifest.style_guide}")
    
    print("\n--- Corrections ---")
    for change in result.changes:
        print(f"[{change['id']}] '{change['original']}' -> '{change['corrected']}'")
        
    if "Manchester City" in result.content:
        print("FAILURE: 'Man City' was expanded to 'Manchester City'.")
    else:
        print("SUCCESS: 'Man City' was preserved.")

if __name__ == "__main__":
    asyncio.run(verify())
