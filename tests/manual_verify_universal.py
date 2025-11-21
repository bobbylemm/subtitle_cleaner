import asyncio
import os
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.universal_corrector import UniversalCorrectionService
from app.domain.constants import SubtitleFormat

async def main():
    print("🚀 Starting Universal Corrector Verification...")
    
    # Check API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  WARNING: OPENAI_API_KEY not found in env. This test might fail if not in .env file loaded by config.")
    
    service = UniversalCorrectionService()
    
    # 1. Test with "Mecano" Challenge (Synthetic)
    print("\n🧪 Test 1: Synthetic 'Mecano' Challenge")
    mecano_srt = """1
00:00:01,000 --> 00:00:05,000
The transfer news is heating up with Mecano moving to Bayern.

2
00:00:05,500 --> 00:00:10,000
And May United are looking to sign a new striker.
"""
    print(f"Input:\n{mecano_srt}")
    
    try:
        result = await service.process_full_document(mecano_srt, SubtitleFormat.SRT)
        print(f"\nOutput:\n{result}")
        
        if "Upamecano" in result and ("Man United" in result or "Manchester United" in result):
            print("✅ PASS: 'Mecano' -> 'Upamecano' and 'May United' -> 'Man United'")
        else:
            print("❌ FAIL: Corrections not found.")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

    # 2. Test with Nigerian Gossip (Synthetic)
    print("\n🧪 Test 2: Nigerian Gossip (Slang Preservation)")
    gossip_srt = """1
00:00:01,000 --> 00:00:05,000
Did you hear what Wizzy said about the concert?

2
00:00:05,500 --> 00:00:10,000
He said the gbedu was too soft, no cap.
"""
    print(f"Input:\n{gossip_srt}")
    
    try:
        result = await service.process_full_document(gossip_srt, SubtitleFormat.SRT)
        print(f"\nOutput:\n{result}")
        
        if "Wizzy" in result and "gbedu" in result:
            print("✅ PASS: Slang 'Wizzy' and 'gbedu' preserved.")
        else:
            print("❌ FAIL: Slang was wrongly corrected.")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())
