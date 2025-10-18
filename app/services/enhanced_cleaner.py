"""Enhanced Subtitle Cleaner - Integration of Layers 3-6"""

import logging
import asyncio
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

from app.services.cleaner import SubtitleCleaner
from app.services.smart_entity_matcher import SmartEntityMatcher
from app.services.ml_context_corrector import get_ml_corrector, CorrectionMode
from app.services.context_extraction_improved import (
    ImprovedContextExtractor as EnhancedContextExtractor,
    ContextSource, 
    SourceType,
    ExtractedEntity
)
from app.services.retrieval_engine import (
    OnDemandRetriever,
    RetrievalResult,
    RetrievalDecision
)
from app.services.llm_selector import (
    LLMSelector,
    SelectionResult,
    AmbiguousEntity
)
from app.services.tenant_memory import (
    TenantMemory,
    CorrectionSource,
    ConfidenceLevel
)
from app.services.tenant_memory_pg import TenantMemoryPostgreSQL

logger = logging.getLogger(__name__)


@dataclass
class EnhancedCleaningConfig:
    """Configuration for enhanced cleaning features"""
    
    # Layer 3: Context Extraction
    enable_context_extraction: bool = True
    context_sources: List[ContextSource] = field(default_factory=list)
    
    # Layer 4: Retrieval
    enable_retrieval: bool = True
    max_retrieval_spans: int = 5
    region_hint: Optional[str] = None
    
    # Layer 5: LLM Selection
    enable_llm_selection: bool = False  # Disabled by default (requires API key)
    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-3.5-turbo"
    
    # Layer 6: Tenant Memory
    enable_tenant_memory: bool = True
    tenant_id: Optional[str] = None
    min_confidence_apply: ConfidenceLevel = ConfidenceLevel.MEDIUM
    use_postgresql: bool = True  # Use PostgreSQL by default
    
    # Auto-context generation
    context_mode: str = "none"  # none, auto, manual, hybrid, smart
    auto_context_options: Optional[Dict] = None
    
    # Contextual correction engine
    correction_mode: str = "balanced"  # legacy, conservative, balanced, aggressive
    
    # ML-based correction
    enable_ml_correction: bool = True
    ml_correction_mode: str = "balanced"  # fast, balanced, or quality
    
    # General
    cache_ttl: int = 900  # 15 minutes
    parallel_processing: bool = True


class EnhancedSubtitleCleaner:
    """
    Enhanced subtitle cleaner integrating all 6 layers:
    - Layer 1-2: Basic cleaning (from original cleaner)
    - Layer 3: Context extraction from user sources
    - Layer 4: On-demand retrieval for suspicious entities
    - Layer 5: LLM selection for ambiguous cases
    - Layer 6: Per-tenant memory and learning
    """
    
    def __init__(self, config: Optional[EnhancedCleaningConfig] = None, db_session=None):
        self.config = config or EnhancedCleaningConfig()
        self.db_session = db_session
        
        # Initialize base cleaner (Layers 1-2)
        self.base_cleaner = SubtitleCleaner()
        
        # Initialize Layer 3: Context Extraction
        self.context_extractor = EnhancedContextExtractor(
            cache_ttl=self.config.cache_ttl
        ) if self.config.enable_context_extraction else None
        
        # Initialize Auto-Context Manager
        from app.services.statistical_context_extractor import AutoContextManager
        self.auto_context_manager = AutoContextManager()
        
        # Initialize Layer 4: Retrieval
        self.retriever = OnDemandRetriever(
            max_spans_per_doc=self.config.max_retrieval_spans,
            cache_ttl=self.config.cache_ttl
        ) if self.config.enable_retrieval else None
        
        # Initialize Layer 5: LLM Selector
        self.llm_selector = LLMSelector(
            api_key=self.config.openai_api_key,
            model=self.config.llm_model,
            cache_ttl=self.config.cache_ttl
        ) if self.config.enable_llm_selection and self.config.openai_api_key else None
        
        # Initialize Layer 6: Tenant Memory
        if self.config.enable_tenant_memory:
            if self.config.use_postgresql and db_session:
                self.tenant_memory_pg = TenantMemoryPostgreSQL()
                self.tenant_memory = None
                logger.info("Using PostgreSQL for tenant memory")
            else:
                self.tenant_memory = TenantMemory(
                    redis_client=None,  # Will use file storage
                    storage_path=Path("./tenant_data")
                )
                self.tenant_memory_pg = None
                logger.info("Using file storage for tenant memory")
        else:
            self.tenant_memory = None
            self.tenant_memory_pg = None
        
        logger.info(f"Enhanced cleaner initialized with layers: "
                   f"Context={self.config.enable_context_extraction}, "
                   f"Retrieval={self.config.enable_retrieval}, "
                   f"LLM={self.config.enable_llm_selection}, "
                   f"Memory={self.config.enable_tenant_memory}")
    
    async def clean_subtitles(
        self,
        segments: List[Dict],
        language: str = "en",
        tenant_id: Optional[str] = None
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Main entry point for enhanced subtitle cleaning
        
        Returns:
            - Cleaned segments
            - Metadata about cleaning process
        """
        print(f"DEBUG: EnhancedSubtitleCleaner.clean_subtitles called with {len(segments)} segments")
        print(f"DEBUG: Config - enable_context: {self.config.enable_context_extraction}, sources: {len(self.config.context_sources) if self.config.context_sources else 0}")
        
        metadata = {
            "layers_applied": [],
            "corrections_made": {},
            "entities_extracted": 0,
            "retrievals_performed": 0,
            "llm_selections": 0,
            "tenant_corrections": 0
        }
        
        try:
            # Step 1: Apply basic cleaning (Layers 1-2) - skipped since already done in router
            # segments = await self._apply_basic_cleaning(segments, language)
            metadata["layers_applied"].extend(["layer1_hygiene", "layer2_stabilization"])
            
            # Step 2: Handle context generation based on mode
            context_sources = self.config.context_sources or []
            context_lexicon = {}  # Initialize early for auto-context
            
            # Apply auto-context generation if enabled
            if self.config.context_mode != "none":
                print(f"DEBUG: Auto-context mode enabled: {self.config.context_mode}")
                from app.services.statistical_context_extractor import ContextMode, StatisticalContextExtractor
                from app.services.context_extraction_improved import ExtractedEntity
                
                # Get full document text for analysis
                document_text = ' '.join(seg.get('text', '') for seg in segments)
                
                # Generate context based on mode
                mode = ContextMode(self.config.context_mode)
                
                # For auto mode, directly extract entities instead of creating sources
                if mode == ContextMode.AUTO:
                    # Direct entity extraction
                    stat_extractor = StatisticalContextExtractor()
                    result = stat_extractor.extract(document_text, self.config.auto_context_options or {})
                    
                    if result.entities:
                        # Convert to ExtractedEntity format for compatibility
                        for entity_text, confidence in result.entities.items():
                            extracted_entity = ExtractedEntity(
                                text=entity_text,
                                canonical_form=entity_text,  # Use the entity text as canonical form
                                entity_type="auto_detected",
                                confidence=confidence,
                                source_id="auto_generated",
                                context=f"Automatically extracted from document (domain: {result.domain})"
                            )
                            context_lexicon[entity_text] = extracted_entity
                        
                        metadata["auto_context"] = {
                            "source": "auto_generated",
                            "entities_found": len(result.entities),
                            "confidence": result.confidence,
                            "domain": result.domain
                        }
                        
                        print(f"DEBUG: Auto-extracted {len(result.entities)} entities directly")
                        print(f"DEBUG: Auto entities include: Man United={('Man United' in result.entities)}, Upamecano={('Upamecano' in result.entities)}")
                        logger.info(f"Auto-extracted entities: {list(result.entities.keys())[:10]}")
                else:
                    # Use the standard flow for other modes
                    auto_sources, auto_metadata = await self.auto_context_manager.generate_context(
                        content=document_text,
                        mode=mode,
                        user_sources=context_sources,
                        options=self.config.auto_context_options
                    )
                    
                    # Update context sources
                    if auto_sources:
                        context_sources = auto_sources
                        metadata["auto_context"] = auto_metadata
                        print(f"DEBUG: Auto-generated {len(auto_sources)} context sources")
                        logger.info(f"Auto-generated context with mode '{mode.value}': {auto_metadata}")
                        
                        # Update config to use the generated sources
                        self.config.context_sources = context_sources
                    else:
                        print(f"DEBUG: No auto-context sources generated")
            
            # Step 2b: Extract context from sources (manual or auto-generated)
            # Note: context_lexicon may already have entities from auto mode
            if self.config.enable_context_extraction and context_sources:
                print(f"DEBUG: Extracting context from {len(context_sources)} sources")
                try:
                    # Extract from sources and merge with any auto-extracted entities
                    additional_lexicon = await self._extract_context()
                    context_lexicon.update(additional_lexicon)
                    print(f"DEBUG: Extracted {len(context_lexicon)} entities")
                    if len(context_lexicon) > 0:
                        # Use list() to avoid dictionary iteration issues
                        items = list(context_lexicon.items())
                        for key, entity in items[:10]:  # Show first 10
                            print(f"DEBUG: Entity: key='{key}', text='{entity.text}', confidence={entity.confidence:.2f}")
                    else:
                        print(f"DEBUG: No entities extracted - checking extractor type")
                        print(f"DEBUG: Using extractor: {type(self.context_extractor).__name__}")
                except Exception as e:
                    print(f"DEBUG: Context extraction failed: {e}")
                    import traceback
                    traceback.print_exc()
                metadata["entities_extracted"] = len(context_lexicon)
                metadata["layers_applied"].append("layer3_context")
                logger.info(f"Extracted {len(context_lexicon)} entities from context: {list(context_lexicon.keys())[:5]}")
            
            # Step 3: Apply tenant corrections if available (Layer 6 - first pass)
            if self.config.enable_tenant_memory and tenant_id:
                segments, tenant_corrections = await self._apply_tenant_corrections(
                    segments, tenant_id, ConfidenceLevel.HIGH
                )
                metadata["tenant_corrections"] = len(tenant_corrections)
                if tenant_corrections:
                    metadata["corrections_made"].update(tenant_corrections)
                    metadata["layers_applied"].append("layer6_memory_apply")
            
            # Step 4: Perform retrieval for suspicious entities (Layer 4)
            retrieval_results = []
            if self.config.enable_retrieval:
                retrieval_results = await self._perform_retrieval(segments)
                metadata["retrievals_performed"] = len(retrieval_results)
                if retrieval_results:
                    metadata["layers_applied"].append("layer4_retrieval")
            
            # Step 5: Apply high-confidence corrections from retrieval
            if retrieval_results:
                segments = self._apply_retrieval_corrections(segments, retrieval_results)
                
                # Learn from high-confidence retrievals (Layer 6)
                if self.config.enable_tenant_memory and tenant_id:
                    await self._learn_from_retrievals(tenant_id, retrieval_results)
            
            # Step 6: LLM selection for ambiguous cases (Layer 5)
            if self.llm_selector and retrieval_results:
                ambiguous_entities = self._identify_ambiguous_entities(retrieval_results)
                
                if ambiguous_entities:
                    selection_results = await self._perform_llm_selection(
                        ambiguous_entities,
                        retrieval_results,
                        list(context_lexicon.keys())
                    )
                    
                    metadata["llm_selections"] = len(selection_results)
                    
                    # Apply LLM selections
                    segments = self._apply_llm_selections(segments, selection_results)
                    
                    # Learn from LLM selections (Layer 6)
                    if self.config.enable_tenant_memory and tenant_id:
                        await self._learn_from_selections(tenant_id, selection_results)
                    
                    metadata["layers_applied"].append("layer5_llm")
            
            # Step 7: Apply context-based corrections
            if context_lexicon:
                print(f"DEBUG: Applying context corrections with {len(context_lexicon)} entities")
                print(f"DEBUG: Context entities: {list(context_lexicon.keys())[:10]}")
                print(f"DEBUG: Has Man United: {'Man United' in context_lexicon}, Has Upamecano: {'Upamecano' in context_lexicon}")
                segments, applied_corrections = self._apply_context_corrections(segments, context_lexicon)
                print(f"DEBUG: Context corrections applied: {applied_corrections}")
                
                # Learn from applied corrections (Layer 6)
                if self.config.enable_tenant_memory and tenant_id and applied_corrections:
                    await self._learn_from_applied_corrections(tenant_id, applied_corrections)
            
            # Final: Learn any remaining corrections for future use
            if self.config.enable_tenant_memory and tenant_id:
                metadata["layers_applied"].append("layer6_memory_learn")
            
            return segments, metadata
            
        except Exception as e:
            logger.error(f"Enhanced cleaning error: {e}")
            # Fallback - return segments unchanged with error metadata
            return segments, {
                "error": str(e),
                "layers_applied": ["layer1_hygiene", "layer2_stabilization"]
            }
    
    async def _apply_basic_cleaning(
        self,
        segments: List[Dict],
        language: str
    ) -> List[Dict]:
        """Apply Layers 1-2: Basic cleaning and stabilization"""
        
        # Base cleaner expects a document, but we're working with segment dicts
        # For now, skip base cleaning when using enhanced mode
        # The basic cleaning is already applied before enhanced features in the router
        return segments
    
    async def _extract_context(self) -> Dict[str, ExtractedEntity]:
        """Extract context from user-provided sources (Layer 3)"""
        
        if not self.context_extractor or not self.config.context_sources:
            return {}
        
        try:
            lexicon = await self.context_extractor.extract_context(
                self.config.context_sources
            )
            
            logger.info(f"Extracted {len(lexicon)} entities from {len(self.config.context_sources)} sources")
            return lexicon
            
        except Exception as e:
            logger.error(f"Context extraction failed: {e}")
            return {}
    
    async def _perform_retrieval(
        self,
        segments: List[Dict]
    ) -> List[RetrievalResult]:
        """Perform on-demand retrieval for suspicious entities (Layer 4)"""
        
        if not self.retriever:
            return []
        
        try:
            results = await self.retriever.process_document(
                segments,
                region_hint=self.config.region_hint,
                enable_retrieval=True
            )
            
            logger.info(f"Retrieved information for {len(results)} suspicious entities")
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    async def _perform_llm_selection(
        self,
        ambiguous_entities: List[AmbiguousEntity],
        retrieval_results: List[RetrievalResult],
        context_entities: List[str]
    ) -> List[SelectionResult]:
        """Use LLM for ambiguous entity selection (Layer 5)"""
        
        if not self.llm_selector:
            return []
        
        try:
            # Convert retrieval results to dict format
            retrieval_dict = [
                {
                    "original": r.original,
                    "consensus": r.consensus,
                    "candidates": [{"text": c.text} for c in r.candidates]
                }
                for r in retrieval_results
            ]
            
            results = await self.llm_selector.process_ambiguous_entities(
                ambiguous_entities,
                retrieval_dict,
                context_entities,
                region_hint=self.config.region_hint
            )
            
            logger.info(f"LLM selected corrections for {len(results)} ambiguous entities")
            return results
            
        except Exception as e:
            logger.error(f"LLM selection failed: {e}")
            return []
    
    async def _apply_tenant_corrections(
        self,
        segments: List[Dict],
        tenant_id: str,
        min_confidence: ConfidenceLevel
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """Apply tenant-specific corrections (Layer 6)"""
        
        if not self.tenant_memory and not self.tenant_memory_pg:
            return segments, {}
        
        try:
            if self.tenant_memory_pg and self.db_session:
                return await self.tenant_memory_pg.apply_tenant_corrections(
                    self.db_session,
                    tenant_id,
                    segments,
                    min_confidence
                )
            elif self.tenant_memory:
                return await self.tenant_memory.apply_tenant_corrections(
                    tenant_id,
                    segments,
                    min_confidence
                )
            else:
                return segments, {}
        except Exception as e:
            logger.error(f"Tenant correction failed: {e}")
            return segments, {}
    
    def _apply_retrieval_corrections(
        self,
        segments: List[Dict],
        retrieval_results: List[RetrievalResult]
    ) -> List[Dict]:
        """Apply high-confidence corrections from retrieval"""
        
        # Build correction map
        corrections = {}
        for result in retrieval_results:
            if result.decision == RetrievalDecision.APPLY and result.consensus:
                corrections[result.original] = result.consensus
        
        if not corrections:
            return segments
        
        # Apply corrections
        corrected_segments = []
        for segment in segments:
            text = segment.get('text', '')
            
            for original, corrected in corrections.items():
                text = text.replace(original, corrected)
            
            segment_copy = segment.copy()
            segment_copy['text'] = text
            corrected_segments.append(segment_copy)
        
        return corrected_segments
    
    def _apply_llm_selections(
        self,
        segments: List[Dict],
        selection_results: List[SelectionResult]
    ) -> List[Dict]:
        """Apply corrections from LLM selection"""
        
        # Build correction map
        corrections = {}
        for result in selection_results:
            if result.confidence >= 0.7:  # Threshold for application
                corrections[result.original] = result.selected
        
        if not corrections:
            return segments
        
        # Apply corrections
        corrected_segments = []
        for segment in segments:
            text = segment.get('text', '')
            
            for original, corrected in corrections.items():
                text = text.replace(original, corrected)
            
            segment_copy = segment.copy()
            segment_copy['text'] = text
            corrected_segments.append(segment_copy)
        
        return corrected_segments
    
    def _apply_context_corrections(
        self,
        segments: List[Dict],
        context_lexicon: Dict[str, ExtractedEntity]
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """Apply context-based corrections using smart entity matching."""
        
        if not context_lexicon:
            logger.info("No context entities to apply")
            return segments
        
        # Create SmartEntityMatcher instance
        try:
            matcher = SmartEntityMatcher()
            # Use smart matching method if available
            if hasattr(matcher, 'find_smart_corrections'):
                # Extract full document text
                document_text = ' '.join(seg.get('text', '') for seg in segments)
                
                print(f"DEBUG: Using SmartEntityMatcher with {len(context_lexicon)} context entities")
                print(f"DEBUG: Context entities: {[entity.text for entity in context_lexicon.values()]}")
                
                # Find corrections using smart matching with correction mode
                correction_mode = getattr(self.config, 'correction_mode', 'balanced')
                corrections = matcher.find_smart_corrections(
                    document_text, 
                    context_lexicon, 
                    correction_mode=correction_mode
                )
            else:
                # Fallback to old method
                document_text = ' '.join(seg.get('text', '') for seg in segments)
                corrections = matcher.find_corrections(document_text, context_lexicon)
        except Exception as e:
            logger.error(f"Failed to initialize matcher: {e}")
            import traceback
            traceback.print_exc()
            return segments, {}
        
        if not corrections:
            print(f"DEBUG: No corrections found by matcher")
            return segments, {}
        
        print(f"DEBUG: Matcher found {len(corrections)} corrections")
        print(f"DEBUG: Corrections: {corrections}")
        
        # Use ML-based correction if enabled
        if self.config.enable_ml_correction:
            print(f"DEBUG: Using ML-based correction with mode: {self.config.ml_correction_mode}")
            ml_corrector = get_ml_corrector(self.config.ml_correction_mode)
            ml_corrected_segments = []
            
            for segment in segments:
                text = segment.get('text', '')
                if text.strip():
                    # Apply ML corrections
                    corrected_text, ml_corrections = ml_corrector.correct_text(text, context_lexicon)
                    
                    if ml_corrections:
                        print(f"DEBUG: ML corrections in segment {segment.get('idx', '?')}: {ml_corrections}")
                    
                    ml_corrected_segments.append({
                        **segment,
                        'text': corrected_text,
                        'ml_corrected': text != corrected_text
                    })
                else:
                    ml_corrected_segments.append(segment)
            
            # Use ML-corrected segments as base for further corrections
            segments = ml_corrected_segments
        
        # Apply corrections to segments with word boundary checking
        corrected_segments = []
        total_corrections_applied = 0
        
        for segment in segments:
            text = segment.get('text', '')
            original_text = text
            corrections_in_segment = 0
            
            # Apply corrections with word boundaries
            # Sort corrections by length (longest first) to prevent partial replacements
            sorted_corrections = sorted(corrections.items(), key=lambda x: len(x[0]), reverse=True)
            
            for incorrect, correct in sorted_corrections:
                # Use word boundary regex for accurate replacement
                pattern = re.compile(r'\b' + re.escape(incorrect) + r'\b', re.IGNORECASE)
                
                # Check if pattern exists in text
                if pattern.search(text):
                    # Replace with correct capitalization
                    new_text = pattern.sub(correct, text)
                    if new_text != text:
                        text = new_text
                        corrections_in_segment += 1
                        print(f"DEBUG: Corrected in segment {segment.get('idx', '?')}: '{incorrect}' -> '{correct}'")
            
            if corrections_in_segment > 0:
                total_corrections_applied += corrections_in_segment
            
            corrected_segments.append({
                **segment,
                'text': text,
                'context_corrected': text != original_text
            })
        
        print(f"DEBUG: Total corrections applied: {total_corrections_applied} in {sum(1 for s in corrected_segments if s.get('context_corrected'))} segments")
        
        return corrected_segments, corrections
    
    def _is_likely_misrecognition(self, text1: str, text2: str) -> bool:
        """
        Check if text1 is likely a misrecognition of text2.
        Handles common ASR errors like "Opus Odima" -> "Hope Uzodinma"
        """
        # Known misrecognition patterns
        known_pairs = [
            ("opus odima", "hope uzodinma"),
            ("opus", "hope"),
            ("odima", "uzodinma"),
            # Add more known pairs as discovered
        ]
        
        t1_lower = text1.lower()
        t2_lower = text2.lower()
        
        for wrong, correct in known_pairs:
            if wrong in t1_lower and correct in t2_lower:
                return True
            if correct in t1_lower and wrong in t2_lower:
                return True
        
        # Check if they have similar structure (same number of words, similar length)
        words1 = t1_lower.split()
        words2 = t2_lower.split()
        
        if len(words1) == len(words2):
            # Check if first letters match (common in ASR errors)
            if all(w1[0] == w2[0] for w1, w2 in zip(words1, words2) if w1 and w2):
                # And lengths are similar
                if all(abs(len(w1) - len(w2)) <= 3 for w1, w2 in zip(words1, words2)):
                    return True
        
        return False
    
    def _identify_ambiguous_entities(
        self,
        retrieval_results: List[RetrievalResult]
    ) -> List[AmbiguousEntity]:
        """Identify entities needing LLM disambiguation"""
        
        if not self.llm_selector:
            return []
        
        ambiguous = []
        for result in retrieval_results:
            if result.decision == RetrievalDecision.SUGGEST:
                # Convert to AmbiguousEntity format
                # This is simplified - would need proper implementation
                from app.services.llm_selector import AmbiguityType
                
                ambiguous.append(AmbiguousEntity(
                    text=result.original,
                    segment_idx=0,  # Would need proper tracking
                    position=0,
                    ambiguity_type=AmbiguityType.SEMANTIC,
                    candidates=[c.text for c in result.candidates[:5]],
                    context="",  # Would need actual context
                    confidence_scores={c.text: c.confidence for c in result.candidates}
                ))
        
        return ambiguous
    
    async def _learn_from_retrievals(
        self,
        tenant_id: str,
        retrieval_results: List[RetrievalResult]
    ):
        """Learn from high-confidence retrieval results"""
        
        if not self.tenant_memory and not self.tenant_memory_pg:
            return
        
        for result in retrieval_results:
            if result.decision == RetrievalDecision.APPLY and result.consensus:
                if self.tenant_memory_pg and self.db_session:
                    await self.tenant_memory_pg.learn_correction(
                        self.db_session,
                        tenant_id,
                        result.original,
                        result.consensus,
                        CorrectionSource.RETRIEVAL,
                        confidence_boost=result.confidence * 0.5
                    )
                elif self.tenant_memory:
                    await self.tenant_memory.learn_correction(
                        tenant_id,
                        result.original,
                        result.consensus,
                        CorrectionSource.RETRIEVAL,
                        confidence_boost=result.confidence * 0.5
                    )
    
    async def _learn_from_selections(
        self,
        tenant_id: str,
        selection_results: List[SelectionResult]
    ):
        """Learn from LLM selections"""
        
        if not self.tenant_memory and not self.tenant_memory_pg:
            return
        
        for result in selection_results:
            if result.confidence >= 0.7:
                if self.tenant_memory_pg and self.db_session:
                    await self.tenant_memory_pg.learn_correction(
                        self.db_session,
                        tenant_id,
                        result.original,
                        result.selected,
                        CorrectionSource.LLM_SELECTION,
                        context=result.reasoning,
                        confidence_boost=result.confidence * 0.3
                    )
                elif self.tenant_memory:
                    await self.tenant_memory.learn_correction(
                        tenant_id,
                        result.original,
                        result.selected,
                        CorrectionSource.LLM_SELECTION,
                        context=result.reasoning,
                        confidence_boost=result.confidence * 0.3
                    )
    
    async def _learn_from_context(
        self,
        tenant_id: str,
        context_lexicon: Dict[str, ExtractedEntity],
        segments: List[Dict]
    ):
        """Learn from context matches found in segments"""
        
        if not self.tenant_memory and not self.tenant_memory_pg:
            return
        
        # Track which entities were actually used
        segment_text = " ".join(s.get('text', '') for s in segments)
        
        for key, entity in context_lexicon.items():
            if entity.text in segment_text and entity.confidence >= 0.7:
                # Learn this correction
                if self.tenant_memory_pg and self.db_session:
                    await self.tenant_memory_pg.learn_correction(
                        self.db_session,
                        tenant_id,
                        key,
                        entity.text,
                        CorrectionSource.CONTEXT_MATCH,
                        context=entity.context,
                        confidence_boost=entity.confidence * 0.4
                    )
                elif self.tenant_memory:
                    await self.tenant_memory.learn_correction(
                        tenant_id,
                        key,
                        entity.text,
                        CorrectionSource.CONTEXT_MATCH,
                        context=entity.context,
                        confidence_boost=entity.confidence * 0.4
                    )
    
    async def _learn_from_applied_corrections(
        self,
        tenant_id: str,
        corrections: Dict[str, str]
    ):
        """Learn from corrections that were actually applied to segments"""
        
        if not self.tenant_memory and not self.tenant_memory_pg:
            return
        
        logger.info(f"Learning {len(corrections)} applied corrections for tenant {tenant_id}")
        
        for original, corrected in corrections.items():
            # Learn each correction with high confidence since it was applied
            if self.tenant_memory_pg and self.db_session:
                await self.tenant_memory_pg.learn_correction(
                    self.db_session,
                    tenant_id,
                    original,
                    corrected,
                    CorrectionSource.CONTEXT_MATCH,
                    context="Applied from context extraction",
                    confidence_boost=0.3  # High boost since it was successfully applied
                )
                logger.info(f"Learned correction in PostgreSQL: '{original}' -> '{corrected}'")
            elif self.tenant_memory:
                await self.tenant_memory.learn_correction(
                    tenant_id,
                    original,
                    corrected,
                    CorrectionSource.CONTEXT_MATCH,
                    context="Applied from context extraction",
                    confidence_boost=0.3
                )
                logger.info(f"Learned correction in file storage: '{original}' -> '{corrected}'")