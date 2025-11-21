import json
import logging
import math
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from app.core.config import settings
from app.services.parser import SubtitleParser
from app.domain.models import SubtitleDocument, Segment
from app.domain.constants import SubtitleFormat
from app.services.knowledge_base import KnowledgeBaseService

logger = logging.getLogger(__name__)

class Entity(BaseModel):
    name: str
    type: str = Field(..., description="Person, Place, Organization, Term, Slang")
    description: Optional[str] = None
    common_misspellings: List[str] = []

class ContextManifest(BaseModel):
    topic: str
    genre: str
    industry: str = "General"
    country: str = "General"
    summary: str
    entities: List[Entity]
    style_guide: str = Field(..., description="Instructions on tone, slang preservation, etc.")
    
    def to_prompt_context(self) -> str:
        """Convert manifest to a concise prompt string."""
        entities_str = "\n".join([f"- {e.name} ({e.type}): {e.description or ''} (Watch out for: {', '.join(e.common_misspellings)})" for e in self.entities])
        return f"""
TOPIC: {self.topic}
GENRE: {self.genre}
INDUSTRY: {self.industry}
COUNTRY: {self.country}
STYLE: {self.style_guide}
KEY ENTITIES:
{entities_str}
"""

class UniversalCorrectionService:
    def __init__(self, api_key: Optional[str] = None, kb_path: str = "knowledge_base.db"):
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            logger.warning("UniversalCorrectionService initialized without API Key. LLM features will fail.")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=self.api_key)
            
        self.model = settings.OPENAI_MODEL or "gpt-3.5-turbo"
        self.kb = KnowledgeBaseService(kb_path)

    async def analyze_context(self, document: SubtitleDocument) -> ContextManifest:
        """
        Stage 1: Analyze the full document to understand context and build a glossary.
        """
        if not self.client:
            logger.error("No OpenAI Client available")
            return ContextManifest(
                topic="General",
                genre="Unknown",
                industry="General",
                country="General",
                summary="Analysis failed - No API Key",
                entities=[],
                style_guide="Correct standard English errors. Preserve intent."
            )

        # Extract a representative sample (first 100 lines + middle 100 + last 100)
        # or just the full text if it's small enough. For now, let's grab a large chunk.
        full_text = "\n".join([s.text for s in document.segments])
        
        # Truncate to avoid token limits if necessary (approx 10k chars for analysis)
        sample_text = full_text[:5000] + "\n...\n" + full_text[-5000:] if len(full_text) > 10000 else full_text

        system_prompt = """
You are a Universal Subtitle Context Analyzer. 
Your goal is to read the raw subtitles and extract a "Context Manifest" to help a corrector fix errors.

1. DETECT TOPIC, GENRE, INDUSTRY, COUNTRY:
   - Topic: Sports, Music, Politics?
   - Industry: Entertainment, Tech, Finance?
   - Country: Where is this taking place? (e.g., "Nigeria", "Germany", "USA"). Crucial for slang/entities.

2. EXTRACT ENTITIES & SLANG:
   - List Names, Places, and specialized Terms.
   - IDENTIFY SLANG/NICKNAMES that should NOT be corrected (e.g., "Wizzy", "Gbedu", "No cap").
   - IDENTIFY PHONETIC ERRORS (e.g., "Mecano" -> "Upamecano", "May United" -> "Man United").

3. DEFINE STYLE GUIDE:
   - "Strict" (Formal news) vs "Loose" (Casual/Slang).
   - Explicitly list words to PRESERVE.

Output JSON:
{
  "topic": "Football",
  "genre": "Analysis",
  "industry": "Sports",
  "country": "Germany",
  "summary": "...",
  "entities": [
    {"name": "Upamecano", "type": "Person", "description": "Bayern Player", "common_misspellings": ["Mecano", "Upamaguire"]}
  ],
  "style_guide": "Preserve all Nigerian slang (Wizzy, gbedu). Fix only obvious phonetic typos."
}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze these subtitles:\n\n{sample_text}"}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            manifest = ContextManifest(**data)
            logger.info(f"Context Analysis Complete: {manifest.topic} ({len(manifest.entities)} entities)")

            return manifest
            
        except Exception as e:
            logger.error(f"Context Analysis Failed: {e}")
            # Return a safe fallback
            return ContextManifest(
                topic="General",
                genre="Unknown",
                industry="General",
                country="General",
                summary=f"Analysis failed: {e}",
                entities=[],
                style_guide="Correct standard English errors. Preserve intent."
            )

    async def correct_chunk(self, segments: List[Segment], manifest: ContextManifest) -> List[Segment]:
        """
        Stage 2: Correct a chunk of segments using the manifest.
        """
        if not segments:
            return []
            
        if not self.client:
            return segments

        # Format input for LLM
        input_text = ""
        for s in segments:
            input_text += f"[{s.idx}] {s.text}\n"

        # Inject KB corrections into the prompt context
        # Use weighted retrieval based on manifest context
        context_dict = {
            "topic": manifest.topic,
            "industry": manifest.industry,
            "country": manifest.country
        }
        
        # We want to find relevant corrections for words in this chunk.
        # Querying the DB for *every* word is too slow.
        # Strategy: Query for known entities in the manifest + common words if we had a cache.
        # For now, let's fetch ALL corrections for this context (filtered by topic/country) to put in prompt.
        # Since we don't have a "get_all_for_context" yet, let's just get all and filter in python or trust the prompt.
        # Actually, let's just pass the most relevant ones.
        
        all_kb = self.kb.get_all_corrections()
        # Filter manually for relevance to this chunk? No, that's hard.
        # Let's just dump the high-confidence ones that match the context.
        
        relevant_kb = []
        for entry in all_kb:
            # Simple relevance check
            score = 0
            if entry.country.lower() == manifest.country.lower(): score += 2
            if entry.topic.lower() == manifest.topic.lower(): score += 1
            if entry.industry.lower() == manifest.industry.lower(): score += 1
            if entry.country == "general": score += 0.5
            
            if score > 0:
                relevant_kb.append(entry)
                
        # Sort by relevance
        # relevant_kb.sort(key=lambda x: x.confidence, reverse=True) # We don't have score here easily
        
        kb_context = "\n".join([f"- {c.wrong_term} -> {c.correct_term}" for c in relevant_kb[:50]]) # Limit to 50
        
        system_prompt = f"""
You are a World-Class Subtitle Corrector.
Your Goal: Fix phonetic errors and transcription mistakes while PRESERVING the speaker's authentic voice.
Accuracy Target: 99%.

CONTEXT MANIFEST:
{manifest.to_prompt_context()}

KNOWLEDGE BASE (PRIORITY FIXES):
{kb_context}

RULES:
1. USE THE KNOWLEDGE BASE & GLOSSARY: Priority 1.
2. PRESERVE SLANG/NICKNAMES: Do NOT correct "Wizzy" to "Whizzy" or "Gbedu" to "Bed" if the context supports it.
3. FIX OBVIOUS TYPOS: "teh" -> "the", "wanna" -> "want to" (ONLY if formal).
4. CONSISTENCY: Use "Man United" (not "Manchester United" unless spoken). Use "Frenkie de Jong" (not "Frankie" or "The Young"). Use "Xavi" (not "Chavi" or "Xvi").
5. DO NOT HALLUCINATE: If unsure, keep the original text.
6. OUTPUT JSON: List of corrections.

Example Output:
{{
  "corrections": [
    {{"idx": 1, "text": "Corrected text line 1"}}
  ]
}}
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Correct these subtitles:\n\n{input_text}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            corrections = {c['idx']: c['text'] for c in data.get('corrections', [])}
            
            # Apply corrections
            corrected_segments = []
            for s in segments:
                new_text = corrections.get(s.idx, s.text) # Fallback to original if no correction returned
                corrected_segments.append(Segment(
                    idx=s.idx,
                    start_ms=s.start_ms,
                    end_ms=s.end_ms,
                    text=new_text
                ))
            
            return corrected_segments

        except Exception as e:
            logger.error(f"Chunk Correction Failed: {e}")
            return segments # Fallback to original

    async def process_full_document(self, content: str, format: SubtitleFormat = SubtitleFormat.SRT) -> str:
        """
        Orchestrate the full correction process.
        """
        # 1. Parse
        doc = SubtitleParser.parse(content, format)
        if not doc.segments:
            return content

        # 2. Analyze Context (The "Brain")
        manifest = await self.analyze_context(doc)
        
        # 3. Chunk and Correct (The "Editor")
        # Chunk size of 20-30 segments is usually good for LLM context window vs latency
        CHUNK_SIZE = 25
        corrected_segments = []
        
        for i in range(0, len(doc.segments), CHUNK_SIZE):
            chunk = doc.segments[i : i + CHUNK_SIZE]
            corrected_chunk = await self.correct_chunk(chunk, manifest)
            corrected_segments.extend(corrected_chunk)
            
        # 4. Global Consistency Pass (The "Polisher")
        corrected_segments = self._apply_global_consistency(corrected_segments, manifest)

        # 5. Reconstruct (Simple serialization)
        output = []
        for s in corrected_segments:
            start = self._format_time(s.start_ms)
            end = self._format_time(s.end_ms)
            output.append(f"{s.idx}\n{start} --> {end}\n{s.text}\n")
            
        return "\n".join(output)

    def _apply_global_consistency(self, segments: List[Segment], manifest: ContextManifest) -> List[Segment]:
        """
        Enforce global consistency for key entities.
        """
        # Start with hardcoded replacements
        # In a real system, this would be more sophisticated (regex, token matching)
        replacements = {
            "Bentancour": "Bentancur",
            "Bentancurt": "Bentancur",
            "The Young": "De Jong",
            "Frankie de Jong": "Frenkie de Jong",
            "Frankie": "Frenkie", 
            "Chavi": "Xavi",
            "Xvi": "Xavi",
            "Manchester United": "Man United",
            "May United": "Man United",
            "M United": "Man United",
            "Superano": "Upamecano",
            "Uruguai": "Uruguay"
        }
        
        # Add entities from manifest to replacements if they have common misspellings
        for entity in manifest.entities:
            for misspelling in entity.common_misspellings:
                replacements[misspelling] = entity.name
                
        # Add entities from Knowledge Base (The most important part!)
        # We fetch ALL corrections because this is a cheap string replacement pass
        all_kb = self.kb.get_all_corrections()
        for entry in all_kb:
            replacements[entry.wrong_term] = entry.correct_term

        final_segments = []
        for s in segments:
            text = s.text
            for wrong, right in replacements.items():
                # Case-insensitive replacement could be better, but let's stick to exact for now to avoid accidents
                if wrong in text:
                    text = text.replace(wrong, right)
                # Also try case-insensitive for specific known bads
                if wrong.lower() in text.lower() and wrong.lower() not in ["may", "man"]: # Avoid common words
                     # This is a bit hacky, better to use regex with word boundaries
                     import re
                     pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                     text = pattern.sub(right, text)

            final_segments.append(Segment(
                idx=s.idx,
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                text=text
            ))
        return final_segments

    def _format_time(self, ms: int) -> str:
        """Format milliseconds to SRT timestamp (00:00:00,000)"""
        seconds, milliseconds = divmod(ms, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
