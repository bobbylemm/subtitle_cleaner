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
        # Sort entities to ensure deterministic prompt for caching
        sorted_entities = sorted(self.entities, key=lambda e: e.name)
        entities_str = "\n".join([f"- {e.name} ({e.type}): {e.description or ''} (Watch out for: {', '.join(e.common_misspellings)})" for e in sorted_entities])
        return f"""
TOPIC: {self.topic}
GENRE: {self.genre}
INDUSTRY: {self.industry}
COUNTRY: {self.country}
STYLE: {self.style_guide}
KEY ENTITIES:
{entities_str}
"""

class Correction(BaseModel):
    segment_id: int
    original_text: str
    corrected_text: str
    type: str = Field(..., description="entity, context, grammar, hallucination")
    reason: str

class CorrectionResult(BaseModel):
    content: str
    manifest: ContextManifest
    changes: List[Dict[str, Any]]

class UniversalCorrectionService:
    def __init__(self, api_key: Optional[str] = None, kb_path: str = "knowledge_base.db"):
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            logger.warning("UniversalCorrectionService initialized without API Key. LLM features will fail.")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=self.api_key)
            
        self.model = settings.OPENAI_MODEL or "gpt-4o"
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
        Stage 2: Correct a chunk of segments using a Deterministic Diff-Based approach.
        """
        if not segments:
            return []
            
        if not self.client:
            return segments

        # Format input for LLM
        input_text = ""
        for s in segments:
            input_text += f"[{s.idx}] {s.text}\n"

        # Fetch relevant KB entries
        all_kb = self.kb.get_all_corrections()
        relevant_kb = []
        for entry in all_kb:
            score = 0
            if entry.country.lower() == manifest.country.lower(): score += 2
            if entry.topic.lower() == manifest.topic.lower(): score += 1
            if entry.industry.lower() == manifest.industry.lower(): score += 1
            if entry.country == "general": score += 0.5
            
            if score > 0:
                relevant_kb.append(entry)
        
        # Sort KB entries to ensure deterministic prompt for caching
        relevant_kb.sort(key=lambda x: x.wrong_term)
        
        kb_context = "\n".join([f"- {c.wrong_term} -> {c.correct_term}" for c in relevant_kb[:50]])
        
        system_prompt = f"""
You are a World-Class Subtitle Corrector.
Your Goal: Identify errors in the subtitles and provide specific corrections.

CONTEXT MANIFEST:
{manifest.to_prompt_context()}

KNOWLEDGE BASE (PRIORITY FIXES):
{kb_context}

INSTRUCTIONS:
1. Analyze the text for errors.
2. Do NOT rewrite the text. Only list specific corrections.
3. Return a JSON object with a "corrections" list.

ERROR CLASSES:
- "entity": Names, Places, Terms (e.g., "Mecano" -> "Upamecano").
- "context": Homophones, Wrong Word (e.g., "contrast" -> "contract"). MUST include surrounding words in 'original_text' to be unique (e.g. "the contrast was").
- "grammar": Punctuation, Casing (e.g., "lets" -> "Let's").
- "hallucination": Text that shouldn't be there (e.g., "Thank you for watching").

RULES:
1. "original_text" MUST MATCH the source text EXACTLY. If it doesn't match, the correction will be rejected.
2. For "context" errors, include 1-2 surrounding words in "original_text" to ensure uniqueness.
3. PRESERVE SLANG: Do not correct "Wizzy" or "Gbedu" if the style guide allows it.
4. "segment_id" must match the ID in the input.

Example Output:
{{
  "corrections": [
    {{
      "segment_id": 12,
      "original_text": "the contrast was",
      "corrected_text": "the contract was",
      "type": "context",
      "reason": "Homophone error"
    }}
  ]
}}
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Find errors in these subtitles:\n\n{input_text}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Parse corrections
            corrections_data = data.get('corrections', [])
            corrections = []
            for c in corrections_data:
                try:
                    corrections.append(Correction(**c))
                except Exception as e:
                    logger.warning(f"Skipping invalid correction format: {c} - {e}")

            # Apply corrections deterministically
            corrected_segments = []
            segment_map = {s.idx: s for s in segments}
            
            # Group corrections by segment
            corrections_by_seg = {}
            for c in corrections:
                if c.segment_id not in corrections_by_seg:
                    corrections_by_seg[c.segment_id] = []
                corrections_by_seg[c.segment_id].append(c)
            
            for s in segments:
                new_text = s.text
                if s.idx in corrections_by_seg:
                    # Sort by length of original_text descending to avoid partial replacement issues
                    seg_corrections = sorted(corrections_by_seg[s.idx], key=lambda x: len(x.original_text), reverse=True)
                    
                    for correction in seg_corrections:
                        # STRICT VALIDATION: Text must exist
                        if correction.original_text in new_text:
                            # Safe replace
                            new_text = new_text.replace(correction.original_text, correction.corrected_text)
                            logger.info(f"Applied {correction.type} fix: '{correction.original_text}' -> '{correction.corrected_text}'")
                        else:
                            logger.warning(f"Rejected {correction.type} fix: '{correction.original_text}' not found in segment {s.idx}")
                
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

    async def process_full_document(self, content: str, format: SubtitleFormat = SubtitleFormat.SRT) -> CorrectionResult:
        """
        Orchestrate the full correction process and return rich results.
        """
        # 1. Parse
        doc = SubtitleParser.parse(content, format)
        if not doc.segments:
            return CorrectionResult(content=content, manifest=ContextManifest(topic="Unknown", genre="Unknown", summary="Empty", entities=[], style_guide=""), changes=[])

        # 2. Analyze Context (The "Brain")
        manifest = await self.analyze_context(doc)
        
        # 3. Chunk and Correct (The "Editor")
        CHUNK_SIZE = 30
        corrected_segments = []
        
        for i in range(0, len(doc.segments), CHUNK_SIZE):
            chunk = doc.segments[i : i + CHUNK_SIZE]
            corrected_chunk = await self.correct_chunk(chunk, manifest)
            corrected_segments.extend(corrected_chunk)
            
        # 4. Global Consistency Pass (The "Polisher")
        corrected_segments = self._apply_global_consistency(corrected_segments, manifest)

        # Calculate Diffs for UI
        changes = []
        orig_map = {s.idx: s.text for s in doc.segments}
        for s in corrected_segments:
            original = orig_map.get(s.idx, "")
            if original != s.text:
                changes.append({
                    "id": s.idx,
                    "original": original,
                    "corrected": s.text
                })

        # 5. Reconstruct
        output = []
        for s in corrected_segments:
            start = self._format_time(s.start_ms)
            end = self._format_time(s.end_ms)
            output.append(f"{s.idx}\n{start} --> {end}\n{s.text}\n")
            
        return CorrectionResult(
            content="\n".join(output),
            manifest=manifest,
            changes=changes
        )

    def _apply_global_consistency(self, segments: List[Segment], manifest: ContextManifest) -> List[Segment]:
        """
        Enforce global consistency for key entities.
        """
        # Start with hardcoded replacements
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
                
        # Add entities from Knowledge Base
        all_kb = self.kb.get_all_corrections()
        for entry in all_kb:
            replacements[entry.wrong_term] = entry.correct_term

        final_segments = []
        for s in segments:
            text = s.text
            for wrong, right in replacements.items():
                # Use regex with word boundaries to avoid partial matches (e.g. "in" inside "win")
                import re
                try:
                    pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)
                    text = pattern.sub(right, text)
                except Exception:
                    if wrong in text:
                        text = text.replace(wrong, right)

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
