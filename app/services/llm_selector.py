"""LLM Selector - Layer 5 Implementation"""

import re
import logging
import asyncio
import json
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import openai
from cachetools import TTLCache

logger = logging.getLogger(__name__)


class SelectionMode(Enum):
    DETERMINISTIC = "deterministic"  # Schema-bound selection
    GUIDED = "guided"                # Template-based
    EXTRACTION = "extraction"        # Pattern extraction


class AmbiguityType(Enum):
    PHONETIC = "phonetic"            # Sound-alike entities
    SEMANTIC = "semantic"            # Context-dependent meaning
    ABBREVIATION = "abbreviation"    # Multiple expansions possible
    SPELLING = "spelling"            # Common misspellings
    TITLE = "title"                  # Title/name variations


@dataclass
class AmbiguousEntity:
    text: str
    segment_idx: int
    position: int
    ambiguity_type: AmbiguityType
    candidates: List[str]
    context: str
    confidence_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class SelectionRequest:
    entity: AmbiguousEntity
    retrieval_candidates: List[str]
    context_entities: List[str]
    region_hint: Optional[str] = None
    domain_hint: Optional[str] = None


@dataclass
class SelectionResult:
    original: str
    selected: str
    confidence: float
    reasoning: str
    selection_mode: SelectionMode
    metadata: Dict[str, Any] = field(default_factory=dict)


class SchemaTemplate:
    """Structured templates for deterministic selection"""
    
    PERSON_DISAMBIGUATION = {
        "system_prompt": "You are an entity disambiguation expert. Select the most likely correct entity from candidates based on context. Return only the JSON response.",
        "user_template": """
Given the context and candidates, select the correct entity.

Context: {context}
Original text (possibly incorrect): {original}
Candidates:
{candidates}

Additional context entities: {context_entities}
Region: {region}

Return JSON:
{{
    "selected": "exact text of selected candidate",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}
""",
        "constraints": {
            "max_reasoning_length": 100,
            "must_select_from_candidates": True,
            "confidence_range": [0.0, 1.0]
        }
    }
    
    ABBREVIATION_EXPANSION = {
        "system_prompt": "You are an abbreviation expansion expert. Select the correct expansion based on context.",
        "user_template": """
Expand the abbreviation based on context.

Context: {context}
Abbreviation: {original}
Possible expansions:
{candidates}

Domain: {domain}
Region: {region}

Return JSON:
{{
    "selected": "full expansion text",
    "confidence": 0.0-1.0,
    "reasoning": "why this expansion fits"
}}
""",
        "constraints": {
            "must_be_valid_expansion": True,
            "preserve_original_if_unsure": True
        }
    }
    
    SPELLING_CORRECTION = {
        "system_prompt": "You are a spelling correction expert for transcribed speech.",
        "user_template": """
Correct the spelling based on phonetic similarity and context.

Context: {context}
Possibly misspelled: {original}
Correction candidates:
{candidates}

Return JSON:
{{
    "selected": "corrected spelling",
    "confidence": 0.0-1.0,
    "reasoning": "phonetic/contextual match"
}}
""",
        "constraints": {
            "phonetic_similarity_required": True,
            "context_coherence_required": True
        }
    }


class LLMSelector:
    """Layer 5: Schema-bound LLM selection for ambiguous entities"""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 max_tokens: int = 150,
                 temperature: float = 0.0,  # Deterministic
                 cache_ttl: int = 3600):
        
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache = TTLCache(maxsize=500, ttl=cache_ttl)
        
        # Initialize OpenAI client if API key provided
        if api_key:
            openai.api_key = api_key
            self.llm_available = True
        else:
            self.llm_available = False
            logger.warning("No OpenAI API key provided - using fallback selection")
        
        # Templates for different ambiguity types
        self.templates = {
            AmbiguityType.PHONETIC: SchemaTemplate.PERSON_DISAMBIGUATION,
            AmbiguityType.SEMANTIC: SchemaTemplate.PERSON_DISAMBIGUATION,
            AmbiguityType.ABBREVIATION: SchemaTemplate.ABBREVIATION_EXPANSION,
            AmbiguityType.SPELLING: SchemaTemplate.SPELLING_CORRECTION,
            AmbiguityType.TITLE: SchemaTemplate.PERSON_DISAMBIGUATION
        }
        
        # Confidence thresholds
        self.apply_threshold = 0.85
        self.suggest_threshold = 0.6
    
    async def process_ambiguous_entities(
        self,
        ambiguous_entities: List[AmbiguousEntity],
        retrieval_results: List[Dict],
        context_entities: List[str],
        region_hint: Optional[str] = None,
        enable_llm: bool = True
    ) -> List[SelectionResult]:
        """Main entry point for Layer 5"""
        
        if not ambiguous_entities:
            return []
        
        results = []
        
        # Process each ambiguous entity
        for entity in ambiguous_entities:
            # Build selection request
            request = SelectionRequest(
                entity=entity,
                retrieval_candidates=[r.get('consensus', '') for r in retrieval_results 
                                     if r.get('original') == entity.text and r.get('consensus')],
                context_entities=context_entities[:10],  # Limit context
                region_hint=region_hint,
                domain_hint=self._infer_domain(entity.context)
            )
            
            # Try LLM selection if available
            if self.llm_available and enable_llm:
                result = await self._llm_select(request)
            else:
                # Fallback to rule-based selection
                result = self._fallback_select(request)
            
            results.append(result)
        
        return results
    
    async def _llm_select(self, request: SelectionRequest) -> SelectionResult:
        """Use LLM for schema-bound selection"""
        
        # Check cache
        cache_key = self._get_cache_key(request)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Get appropriate template
            template = self.templates.get(
                request.entity.ambiguity_type,
                SchemaTemplate.PERSON_DISAMBIGUATION
            )
            
            # Format candidates with scores
            candidates_text = self._format_candidates(request)
            
            # Build prompts
            system_prompt = template["system_prompt"]
            user_prompt = template["user_template"].format(
                context=request.entity.context,
                original=request.entity.text,
                candidates=candidates_text,
                context_entities=", ".join(request.context_entities[:5]),
                region=request.region_hint or "unknown",
                domain=request.domain_hint or "general"
            )
            
            # Call OpenAI API
            response = await self._call_openai(system_prompt, user_prompt)
            
            # Parse structured response
            result = self._parse_llm_response(response, request)
            
            # Validate against constraints
            result = self._validate_selection(result, template.get("constraints", {}))
            
            # Cache result
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"LLM selection error: {e}")
            # Fallback to rule-based
            return self._fallback_select(request)
    
    def _fallback_select(self, request: SelectionRequest) -> SelectionResult:
        """Rule-based selection when LLM unavailable"""
        
        candidates = request.entity.candidates + request.retrieval_candidates
        
        if not candidates:
            # No candidates, keep original
            return SelectionResult(
                original=request.entity.text,
                selected=request.entity.text,
                confidence=0.0,
                reasoning="No candidates available",
                selection_mode=SelectionMode.DETERMINISTIC
            )
        
        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            score = self._score_candidate(candidate, request)
            scored_candidates.append((candidate, score))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select best candidate
        best_candidate, best_score = scored_candidates[0]
        
        # Determine confidence based on score distribution
        confidence = self._calculate_confidence(scored_candidates)
        
        return SelectionResult(
            original=request.entity.text,
            selected=best_candidate if confidence >= self.suggest_threshold else request.entity.text,
            confidence=confidence,
            reasoning=f"Fallback selection based on similarity scores",
            selection_mode=SelectionMode.DETERMINISTIC,
            metadata={"scores": dict(scored_candidates[:3])}
        )
    
    def _score_candidate(self, candidate: str, request: SelectionRequest) -> float:
        """Score a candidate based on multiple factors"""
        score = 0.0
        
        # Phonetic similarity for phonetic/spelling ambiguity
        if request.entity.ambiguity_type in [AmbiguityType.PHONETIC, AmbiguityType.SPELLING]:
            import jellyfish
            phonetic_sim = 1.0 if jellyfish.metaphone(candidate) == jellyfish.metaphone(request.entity.text) else 0.3
            score += phonetic_sim * 0.4
        
        # Context entity overlap
        candidate_lower = candidate.lower()
        context_overlap = sum(1 for e in request.context_entities if e.lower() in candidate_lower)
        score += min(context_overlap * 0.1, 0.3)
        
        # Length similarity (for abbreviations)
        if request.entity.ambiguity_type == AmbiguityType.ABBREVIATION:
            if len(candidate) > len(request.entity.text):
                score += 0.3  # Prefer expansions
        
        # Confidence scores if available
        if candidate in request.entity.confidence_scores:
            score += request.entity.confidence_scores[candidate] * 0.3
        
        return min(score, 1.0)
    
    def _calculate_confidence(self, scored_candidates: List[Tuple[str, float]]) -> float:
        """Calculate confidence based on score distribution"""
        if len(scored_candidates) < 2:
            return scored_candidates[0][1] if scored_candidates else 0.0
        
        best_score = scored_candidates[0][1]
        second_score = scored_candidates[1][1]
        
        # High confidence if clear winner
        if best_score > 0.7 and (best_score - second_score) > 0.3:
            return 0.9
        # Medium confidence if decent lead
        elif best_score > 0.5 and (best_score - second_score) > 0.15:
            return 0.7
        # Low confidence if close scores
        else:
            return 0.4
    
    async def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API with retry logic"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    openai.ChatCompletion.create,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                raise e
    
    def _parse_llm_response(self, response: str, request: SelectionRequest) -> SelectionResult:
        """Parse structured JSON response from LLM"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            data = json.loads(json_match.group())
            
            return SelectionResult(
                original=request.entity.text,
                selected=data.get("selected", request.entity.text),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                selection_mode=SelectionMode.GUIDED,
                metadata={"llm_response": data}
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Return original on parse failure
            return SelectionResult(
                original=request.entity.text,
                selected=request.entity.text,
                confidence=0.0,
                reasoning="Failed to parse LLM response",
                selection_mode=SelectionMode.DETERMINISTIC
            )
    
    def _validate_selection(self, result: SelectionResult, constraints: Dict) -> SelectionResult:
        """Validate selection against constraints"""
        
        # Ensure confidence is in valid range
        if "confidence_range" in constraints:
            min_conf, max_conf = constraints["confidence_range"]
            result.confidence = max(min_conf, min(max_conf, result.confidence))
        
        # Truncate reasoning if too long
        if "max_reasoning_length" in constraints:
            max_len = constraints["max_reasoning_length"]
            if len(result.reasoning) > max_len:
                result.reasoning = result.reasoning[:max_len-3] + "..."
        
        return result
    
    def _format_candidates(self, request: SelectionRequest) -> str:
        """Format candidates for prompt"""
        lines = []
        
        # Add entity candidates
        for i, candidate in enumerate(request.entity.candidates[:5], 1):
            score = request.entity.confidence_scores.get(candidate, 0.5)
            lines.append(f"{i}. {candidate} (confidence: {score:.2f})")
        
        # Add retrieval candidates
        for candidate in request.retrieval_candidates[:3]:
            if candidate not in request.entity.candidates:
                lines.append(f"- {candidate} (from external sources)")
        
        return "\n".join(lines)
    
    def _infer_domain(self, context: str) -> str:
        """Infer domain from context"""
        context_lower = context.lower()
        
        if any(word in context_lower for word in ['president', 'governor', 'minister', 'senator']):
            return "politics"
        elif any(word in context_lower for word in ['ceo', 'company', 'business', 'market']):
            return "business"
        elif any(word in context_lower for word in ['professor', 'university', 'research', 'study']):
            return "academic"
        elif any(word in context_lower for word in ['actor', 'movie', 'music', 'artist']):
            return "entertainment"
        else:
            return "general"
    
    def _get_cache_key(self, request: SelectionRequest) -> str:
        """Generate cache key for request"""
        key_parts = [
            request.entity.text,
            request.entity.ambiguity_type.value,
            "|".join(sorted(request.entity.candidates[:5])),
            request.region_hint or "",
            request.domain_hint or ""
        ]
        key_str = ":".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def detect_ambiguous_entities(
        self,
        segments: List[Dict],
        retrieval_results: List[Dict]
    ) -> List[AmbiguousEntity]:
        """Detect entities that need LLM disambiguation"""
        ambiguous = []
        
        for segment in segments:
            text = segment.get('text', '')
            idx = segment.get('idx', 0)
            
            # Check retrieval results for suggestions
            for result in retrieval_results:
                if result.get('decision') == 'SUGGEST':
                    # This entity needs disambiguation
                    candidates = [c.get('text', '') for c in result.get('candidates', [])]
                    
                    ambiguous.append(AmbiguousEntity(
                        text=result.get('original', ''),
                        segment_idx=idx,
                        position=0,  # Would need proper position tracking
                        ambiguity_type=self._detect_ambiguity_type(
                            result.get('original', ''),
                            candidates
                        ),
                        candidates=candidates,
                        context=text,
                        confidence_scores={c: result.get('confidence', 0.5) for c in candidates}
                    ))
        
        return ambiguous
    
    def _detect_ambiguity_type(self, original: str, candidates: List[str]) -> AmbiguityType:
        """Detect type of ambiguity"""
        
        # Check if it's an abbreviation
        if all(len(c) > len(original) for c in candidates):
            return AmbiguityType.ABBREVIATION
        
        # Check if it's phonetic (using simple heuristic)
        import jellyfish
        orig_phonetic = jellyfish.metaphone(original)
        if any(jellyfish.metaphone(c) == orig_phonetic for c in candidates):
            return AmbiguityType.PHONETIC
        
        # Check if it's a title variation
        if any(title in original.lower() for title in ['dr', 'mr', 'mrs', 'prof', 'general']):
            return AmbiguityType.TITLE
        
        # Default to semantic
        return AmbiguityType.SEMANTIC
    
    def build_correction_map(self, selection_results: List[SelectionResult]) -> Dict[str, str]:
        """Build correction map from selection results"""
        corrections = {}
        
        for result in selection_results:
            if result.confidence >= self.apply_threshold:
                # High confidence - apply automatically
                corrections[result.original] = result.selected
            elif result.confidence >= self.suggest_threshold:
                # Medium confidence - mark as suggestion
                corrections[f"[?{result.original}]"] = f"[?{result.selected}]"
        
        return corrections