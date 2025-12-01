import asyncio
import logging
from app.services.universal_corrector import UniversalCorrectionService, Correction, Segment

# Configure logging
logging.basicConfig(level=logging.INFO)

async def reproduce():
    # Mock segment 168
    # 168
    # 00:05:10,160 --> 00:05:12,320
    # his his way, maneuver the space to even
    
    segments = [
        Segment(idx=168, start_ms=0, end_ms=1000, text="his his way, maneuver the space to even")
    ]
    
    # Simulate the correction the LLM is likely generating
    corrections = [
        Correction(
            segment_id=168,
            original_text="his his",
            corrected_text="his",
            type="grammar",
            reason="Repeated word"
        )
    ]
    
    print("\n--- Testing Repeated Word Correction ---")
    print(f"Segment 168: '{segments[0].text}'")
    print(f"Correction: '{corrections[0].original_text}' -> '{corrections[0].corrected_text}'")
    
    # Logic from correct_chunk
    new_text = segments[0].text
    correction = corrections[0]
    
    # Normalize whitespace for check?
    # The current logic is:
    if correction.original_text in new_text:
        new_text = new_text.replace(correction.original_text, correction.corrected_text)
        print(f"SUCCESS: Applied fix: '{new_text}'")
    else:
        print(f"FAILURE: Rejected fix: '{correction.original_text}' not found in segment {168}")
        
        # Check if it's failing due to case sensitivity?
        # Or maybe the LLM output includes punctuation?

if __name__ == "__main__":
    asyncio.run(reproduce())
