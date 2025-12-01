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
    dialect: str = "Standard English"
    summary: str
    entities: List[Entity]
    preservation_list: List[str] = Field(default_factory=list, description="List of slang/dialect terms to preserve exactly.")
    style_guide: str = Field(..., description="Instructions on tone, slang preservation, etc.")
    
    def to_prompt_context(self) -> str:
        """Convert manifest to a concise prompt string."""
        # Sort entities to ensure deterministic prompt for caching
        sorted_entities = sorted(self.entities, key=lambda e: e.name)
        entities_str = "\n".join([f"- {e.name} ({e.type}): {e.description or ''} (Watch out for: {', '.join(e.common_misspellings)})" for e in sorted_entities])
        
        preservation_str = ", ".join(self.preservation_list)
        
        return f"""
TOPIC: {self.topic}
GENRE: {self.genre}
INDUSTRY: {self.industry}
COUNTRY: {self.country}
DIALECT: {self.dialect}
STYLE: {self.style_guide}
PRESERVE TERMS (DO NOT CORRECT): {preservation_str}
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

2. DETECT DIALECT & PRESERVATION LIST (CRITICAL):
   - Identify the dominant dialect (e.g., "Standard English", "Nigerian Pidgin", "AAVE", "Singlish").
   - EXTRACT A "PRESERVATION LIST": A list of specific slang words, interjections, or cultural terms that look like typos but are valid in this dialect.
   - Example for Nigerian Pidgin: ["abeg", "wetin", "na", "sabi", "wahala", "dey", "sef"].
   - Example for AAVE: ["finna", "tryna", "ion"].

3. EXTRACT ENTITIES & SLANG:
   - List Names, Places, and specialized Terms.
   - IDENTIFY PHONETIC ERRORS (e.g., "Mecano" -> "Upamecano").
   - DO NOT list valid abbreviations (e.g., "Man City", "Utd") as misspellings unless the style guide strictly forbids them.

4. DEFINE STYLE GUIDE & NAMING CONVENTIONS:
   - "Strict" (Formal news) vs "Loose" (Casual/Slang).
   - NAMING CONVENTIONS: Does the speaker use full names ("Manchester City") or abbreviations ("Man City")?
   - Explicitly state: "Preserve full names" or "Allow abbreviations" based on the text.

Output JSON:
{
  "topic": "Football",
  "genre": "Analysis",
  "industry": "Sports",
  "country": "Germany",
  "dialect": "Standard English",
  "summary": "...",
  "entities": [
    {"name": "Upamecano", "type": "Person", "description": "Bayern Player", "common_misspellings": ["Mecano", "Upamaguire"]}
  ],
  "preservation_list": [],
  "style_guide": "Formal tone. Speaker uses full club names (e.g., 'Manchester City'). Do NOT shorten proper nouns."
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
            for e in manifest.entities:
                logger.info(f"Entity: {e.name}, Misspellings: {e.common_misspellings}")

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
        logger.info(f"KB Context for chunk: {kb_context}")
        
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
3. STRICTLY FOLLOW THE "NO SHORTENING" RULE. Do NOT abbreviate terms.
4. Return a JSON object with a "corrections" list.

ERROR CLASSES:
- "entity": Names, Places, Terms (e.g., "Mecano" -> "Upamecano").
- "context": Homophones, Wrong Word (e.g., "contrast" -> "contract"). MUST include surrounding words in 'original_text' to be unique (e.g. "the contrast was").
- "grammar": Punctuation, Casing (e.g., "lets" -> "Let's").
- "hallucination": Text that shouldn't be there (e.g., "Thank you for watching").

RULES:
1. "original_text" MUST MATCH the source text EXACTLY. If it doesn't match, the correction will be rejected.
2. For "context" errors, include 1-2 surrounding words in "original_text" to ensure uniqueness.
3. PRESERVE DIALECT: The detected dialect is "{manifest.dialect}". Do NOT correct valid slang or dialect terms (e.g., "{', '.join(manifest.preservation_list[:5])}").
4. NO SHORTENING: DO NOT shorten or abbreviate terms. Keep "Manchester City" as "Manchester City", do NOT change to "Man City".
5. ENTITY FIDELITY: If a proper noun (Person, Place, Org) is already spelled correctly, DO NOT CHANGE IT.
6. VERBATIM PREFERENCE: If the original text is valid, PREFER IT over a synonym or paraphrase.
7. IF UNSURE: If you are unsure if a word is a typo or valid slang, LEAVE IT ALONE.
8. SPLIT SEGMENTS: If an error spans multiple segments (e.g., "Man" in [1] and "City" in [2]), provide SEPARATE corrections for each segment. Do NOT combine them.
9. "segment_id" must match the ID in the input.

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
                            # FALLBACK: Check for split segment error
                            # If the correction spans to the next segment, we need to handle it.
                            next_seg = segment_map.get(s.idx + 1)
                            prev_seg = segment_map.get(s.idx - 1)
                            
                            # 1. Check Forward (Current + Next)
                            if next_seg:
                                # Normalize for check (handle newlines/double spaces)
                                combined_text_norm = " ".join((new_text + " " + next_seg.text).split())
                                original_norm = " ".join(correction.original_text.split())
                                
                                logger.info(f"Checking FORWARD split fallback for '{correction.original_text}' in Seg {s.idx}+{s.idx+1}")
                                logger.info(f"Original Norm: '{original_norm}'")
                                logger.info(f"Combined Norm: '{combined_text_norm}'")
                                
                                if original_norm in combined_text_norm:
                                    logger.warning(f"Detected split segment error for '{correction.original_text}'. Attempting partial fix...")
                                    
                                    # Let's try to match the suffix of Seg 1.
                                    for i in range(len(correction.original_text), 0, -1):
                                        part = correction.original_text[:i]
                                        # Relaxed check: ignore trailing whitespace/newlines in new_text
                                        # Also strip part to avoid matching just spaces
                                        if part.strip() and new_text.rstrip().endswith(part.strip()):
                                            # Found the split point!
                                            remainder = correction.original_text[i:].strip()
                                            
                                            # Check if Seg 2 starts with remainder
                                            if next_seg.text.lstrip().startswith(remainder):
                                                # Apply fix!
                                                match_len = len(part.strip())
                                                stripped_new = new_text.rstrip()
                                                prefix = stripped_new[:-match_len]
                                                
                                                new_text = prefix + correction.corrected_text
                                                
                                                next_seg_stripped = next_seg.text.lstrip()
                                                next_seg.text = next_seg_stripped[len(remainder):].strip()
                                                
                                                logger.info(f"Applied SPLIT fix: '{correction.original_text}' -> '{correction.corrected_text}' (Merged to Seg {s.idx})")
                                                break
                                    else:
                                         logger.warning(f"Rejected {correction.type} fix: '{correction.original_text}' not found in segment {s.idx} (Split detection failed)")
                                         logger.info(f"Failed to find split point. Seg 1 end: '{new_text[-20:]}', Seg 2 start: '{next_seg.text[:20]}'")
                                else:
                                    logger.warning(f"Rejected {correction.type} fix: '{correction.original_text}' not found in segment {s.idx}")
                                    logger.info(f"Combined check failed. Seg {s.idx} text: '{new_text}'")
                            
                            # 2. Check Backward (Previous + Current) - ONLY if Forward failed
                            elif prev_seg:
                                combined_text_norm = " ".join((prev_seg.text + " " + new_text).split())
                                original_norm = " ".join(correction.original_text.split())
                                
                                logger.info(f"Checking BACKWARD split fallback for '{correction.original_text}' in Seg {s.idx-1}+{s.idx}")
                                logger.info(f"Combined Norm: '{combined_text_norm}'")
                                
                                if original_norm in combined_text_norm:
                                     logger.warning(f"Detected BACKWARD split segment error for '{correction.original_text}'.")
                                     # We can't easily fix prev_seg here because we might have already processed it or it's not in our write scope easily?
                                     # Actually, we are modifying `segments` list objects in place via `segment_map`.
                                     # So we CAN modify `prev_seg.text`.
                                     
                                     # Logic similar to forward, but reversed.
                                     # original_text spans Prev + Current.
                                     # We want to find the prefix of Current that matches the suffix of original_text.
                                     
                                     for i in range(len(correction.original_text)):
                                         suffix = correction.original_text[i:]
                                         prefix = correction.original_text[:i]
                                         
                                         if suffix.strip() and new_text.lstrip().startswith(suffix.strip()):
                                             if prev_seg.text.rstrip().endswith(prefix.strip()):
                                                 # Found it!
                                                 # Apply fix to PREV segment (merge there?) or Current?
                                                 # Let's put the fix in Prev segment.
                                                 
                                                 match_len = len(prefix.strip())
                                                 stripped_prev = prev_seg.text.rstrip()
                                                 prev_prefix = stripped_prev[:-match_len]
                                                 
                                                 prev_seg.text = prev_prefix + correction.corrected_text
                                                 
                                                 # Remove from Current
                                                 suffix_len = len(suffix.strip())
                                                 stripped_curr = new_text.lstrip()
                                                 new_text = stripped_curr[suffix_len:].strip()
                                                 
                                                 logger.info(f"Applied BACKWARD SPLIT fix: '{correction.original_text}' -> '{correction.corrected_text}' (Merged to Seg {s.idx-1})")
                                                 break
                                     else:
                                         logger.warning(f"Rejected {correction.type} fix: '{correction.original_text}' (Backward split detection failed)")
                                else:
                                     logger.warning(f"Rejected {correction.type} fix: '{correction.original_text}' not found in segment {s.idx}")
                            else:
                                logger.warning(f"Rejected {correction.type} fix: '{correction.original_text}' not found in segment {s.idx}")
                                logger.info(f"No adjacent segment found for fallback check.")
                
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
            "Superano": "Upamecano",
            "Uruguai": "Uruguay"
        }
        
        # Add entities from manifest to replacements if they have common misspellings
        for entity in manifest.entities:
            for misspelling in entity.common_misspellings:
                # Skip if it looks like a valid abbreviation and style is not Strict
                if len(misspelling) < len(entity.name) and misspelling in entity.name and "Strict" not in manifest.style_guide:
                     continue
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
                    if pattern.search(text):
                        text = pattern.sub(right, text)
                        logger.info(f"Global Consistency Fix: '{wrong}' -> '{right}'")
                except Exception:
                    if wrong in text:
                        text = text.replace(wrong, right)
                        logger.info(f"Global Consistency Fix (Simple): '{wrong}' -> '{right}'")

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
