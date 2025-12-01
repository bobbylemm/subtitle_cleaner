import asyncio
import os
from app.services.universal_corrector import UniversalCorrectionService
from app.domain.constants import SubtitleFormat

from app.core.config import settings

async def verify_file(filename, expected_dialect, expected_terms):
    if not settings.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set in settings")
        return

    service = UniversalCorrectionService()
    
    with open(filename, "r") as f:
        content = f.read()
        
    print(f"\n=== Processing {filename} ===")
    result = await service.process_full_document(content, SubtitleFormat.SRT)
    
    print(f"Detected Dialect: {result.manifest.dialect}")
    print(f"Preservation List: {result.manifest.preservation_list}")
    
    # Validation Logic
    preserved_count = 0
    for term in expected_terms:
        if term.lower() in result.content.lower():
            preserved_count += 1
            
    print(f"Preserved {preserved_count}/{len(expected_terms)} key terms.")
    
    # Check for actual corrections (definately -> definitely)
    if "definately" in content and "definitely" in result.content:
        print("SUCCESS: Typos corrected (definately -> definitely).")
    
    if expected_dialect.lower() in result.manifest.dialect.lower() or result.manifest.dialect.lower() in expected_dialect.lower():
        print(f"Dialect Match: PASSED ({result.manifest.dialect})")
    else:
        print(f"Dialect Match: FAILED (Expected {expected_dialect}, got {result.manifest.dialect})")

async def verify():
    await verify_file("pidgin_sample.srt", "Nigerian Pidgin", ["Abeg", "wetin", "dey", "wahala"])
    await verify_file("british_sample.srt", "British", ["bruv", "innit", "gaffer", "cheeky"])
    await verify_file("jamaican_sample.srt", "Jamaican", ["Wah gwaan", "irie", "pickney"])

if __name__ == "__main__":
    asyncio.run(verify())
