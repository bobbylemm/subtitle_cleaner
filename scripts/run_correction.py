import asyncio
import sys
import os
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.universal_corrector import UniversalCorrectionService
from app.domain.constants import SubtitleFormat

async def main():
    if len(sys.argv) < 3:
        print("Usage: python run_correction.py <input_file> <output_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"🚀 Processing {input_path} -> {output_path}")

    if not os.path.exists(input_path):
        print(f"❌ Error: Input file '{input_path}' not found.")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    service = UniversalCorrectionService()
    
    try:
        # Determine format (simple check)
        fmt = SubtitleFormat.SRT
        if input_path.lower().endswith(".vtt"):
            fmt = SubtitleFormat.WEBVTT

        result = await service.process_full_document(content, fmt)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
            
        print(f"✅ Done! Saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
