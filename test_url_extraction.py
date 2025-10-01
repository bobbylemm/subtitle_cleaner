#!/usr/bin/env python
"""Test URL extraction script"""

import asyncio
import sys
import os
sys.path.insert(0, '/app')

from app.services.context_extraction_improved import ImprovedContextExtractor, ContextSource, SourceType

async def test():
    print("Testing URL extraction...")
    extractor = ImprovedContextExtractor()
    sources = [
        ContextSource(
            source_type=SourceType.URL,
            content='https://dailypost.ng/2025/09/17/jubilation-as-nigerian-army-capture-esn-commander-gentle-de-yahoo/',
            source_id='test',
            authority_score=1.0
        )
    ]
    
    result = await extractor.extract_context(sources)
    print(f'Extracted {len(result)} entities')
    
    if result:
        print("\nEntities found:")
        for i, (key, entity) in enumerate(list(result.items())[:20]):
            print(f'  {i+1}. {entity.text} (type: {entity.entity_type.value}, confidence: {entity.confidence:.2f})')
    else:
        print("No entities extracted - debugging...")
        # Try to fetch the URL manually
        import aiohttp
        from trafilatura import extract
        
        async with aiohttp.ClientSession() as session:
            async with session.get(sources[0].content) as resp:
                print(f"  HTTP Status: {resp.status}")
                if resp.status == 200:
                    html = await resp.text()
                    print(f"  HTML length: {len(html)}")
                    text = extract(html)
                    if text:
                        print(f"  Extracted text length: {len(text)}")
                        print(f"  First 500 chars: {text[:500]}...")
                    else:
                        print("  No text extracted by trafilatura")
                        print(f"  HTML preview: {html[:500]}...")

if __name__ == "__main__":
    asyncio.run(test())