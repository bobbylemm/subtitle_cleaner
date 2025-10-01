import re
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Set
from copy import deepcopy

from app.domain.models import Segment, SubtitleDocument, Settings
from app.domain.constants import (
    MergeMode,
    FillerMode,
    Language,
    FILLERS,
    PUNCTUATION_RULES,
    TAG_PATTERN,
    MULTIPLE_SPACES_PATTERN,
)
# Optional ML imports (gracefully handle missing dependencies)
try:
    from app.ml.base import MLConfig, ModelType
    from app.ml.punctuation import PunctuationModel
    from app.ml.grammar import GrammarCorrectionModel
    from app.ml.filler import ContextualFillerDetector
    from app.ml.scorer import PerplexityScorer
    from app.ml.gatekeeper import Gatekeeper, EditConstraints
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    
from app.ml.punctuation_onnx import ONNXPunctuationModel
from app.services.change_tracker import ChangeTracker, ChangeType, ChangeSource
from app.services.entity_stabilizer import EntityStabilizer
# from app.services.context_extractor import ContextExtractor  # Removed - use enhanced_cleaner for context

logger = logging.getLogger(__name__)

# Phase 2 and 3 removed - keeping only Layer 1-2 features from requirements


class SubtitleCleaner:
    """Main subtitle cleaning service with optional ML enhancement"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.modifications = []
        self.ml_models = {}
        self.gatekeeper = None
        self._ml_initialized = False
        
        # New components for enhanced cleaning
        self.change_tracker = ChangeTracker()
        self.entity_stabilizer = EntityStabilizer()
        # self.context_extractor = ContextExtractor()  # Removed - use enhanced_cleaner for context
        self.onnx_punct_model = None
        self._onnx_initialized = False
        
        # Phase 2 and 3 removed - not part of core requirements
    
    async def initialize_ml(self) -> None:
        """Initialize ML models if enabled"""
        if self._ml_initialized or not getattr(self.settings, 'ml_enabled', False):
            return
            
        if not ML_AVAILABLE:
            logger.warning("ML dependencies not available, skipping ML initialization")
            return
        
        try:
            # Create ML config
            ml_config = MLConfig(
                language=self.settings.language,
                device=getattr(self.settings, 'ml_device', 'cpu'),
                quantized=getattr(self.settings, 'ml_quantized', True),
                cache_size=100
            )
            
            # Initialize models based on settings
            ml_models = getattr(self.settings, 'ml_models', {})
            
            if ml_models.get('punctuation', False):
                self.ml_models['punctuation'] = PunctuationModel(ml_config)
                await self.ml_models['punctuation'].initialize()
            
            if ml_models.get('grammar', False):
                self.ml_models['grammar'] = GrammarCorrectionModel(ml_config)
                await self.ml_models['grammar'].initialize()
            
            if ml_models.get('contextual_fillers', False):
                self.ml_models['filler'] = ContextualFillerDetector(ml_config)
                await self.ml_models['filler'].initialize()
            
            if ml_models.get('scoring', False):
                self.ml_models['scorer'] = PerplexityScorer(ml_config)
                await self.ml_models['scorer'].initialize()
            
            # Initialize gatekeeper
            ml_constraints = getattr(self.settings, 'ml_constraints', {})
            self.gatekeeper = Gatekeeper(
                constraints=EditConstraints(
                    max_edit_ratio=ml_constraints.get('max_edit_ratio', 0.15),
                    max_char_change=ml_constraints.get('max_char_change', 8),
                    min_confidence=ml_constraints.get('min_confidence', 0.7)
                ),
                language=self.settings.language
            )
            
            self._ml_initialized = True
            logger.info(f"ML models initialized: {list(self.ml_models.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            self._ml_initialized = False
    
    async def initialize_onnx_punctuation(self) -> None:
        """Initialize ONNX punctuation model if enabled"""
        if self._onnx_initialized:
            return
            
        if getattr(self.settings, 'enable_punctuation', False):
            try:
                self.onnx_punct_model = ONNXPunctuationModel(
                    language=self.settings.language
                )
                self.onnx_punct_model.initialize()
                self._onnx_initialized = True
                logger.info("ONNX punctuation model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ONNX punctuation: {e}")
                self._onnx_initialized = False
    
    # Phase 2 initialization removed - not part of core requirements
    async def _phase2_placeholder(self) -> None:
        """Placeholder - Phase 2 removed"""
        return
            
        try:
            # Check which Phase 2 features are enabled
            needs_embedder = any([
                getattr(self.settings, 'enable_topic_segmentation', False),
                getattr(self.settings, 'enable_speaker_tracking', False),
                getattr(self.settings, 'enable_coreference_resolution', False)
            ])
            
            # Initialize shared embedder if needed
            if needs_embedder:
                try:
                    from app.ml.semantic_embedder import SemanticEmbedder
                    self.phase2_components['embedder'] = SemanticEmbedder(
                        model_type="minilm",
                        use_cache=True
                    )
                except Exception as e:
                    logger.warning(f"Could not initialize semantic embedder: {e}")
                    # Phase 2 will use fallback methods
            
            # Initialize topic segmentation
            if getattr(self.settings, 'enable_topic_segmentation', False):
                self.phase2_components['topic_segmenter'] = TopicSegmenter(
                    embedder=self.phase2_components.get('embedder'),
                    boundary_threshold=0.6,
                    min_segment_size=3
                )
                logger.info("Topic segmentation initialized")
            
            # Initialize speaker tracking
            if getattr(self.settings, 'enable_speaker_tracking', False):
                self.phase2_components['speaker_tracker'] = LightweightSpeakerTracker(
                    embedder=self.phase2_components.get('embedder'),
                    similarity_threshold=0.55  # Lower threshold for better grouping
                )
                logger.info("Speaker tracking initialized")
            
            # Initialize coreference resolution
            if getattr(self.settings, 'enable_coreference_resolution', False):
                self.phase2_components['coreference_resolver'] = EfficientCoreferenceResolver(
                    embedder=self.phase2_components.get('embedder'),
                    max_distance=5
                )
                logger.info("Coreference resolution initialized")
                
            self._phase2_initialized = True
            logger.info(f"Phase 2 components initialized: {list(self.phase2_components.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Phase 2 components: {e}")
            self._phase2_initialized = False
    
    # Phase 3 initialization removed - not part of core requirements  
    async def _phase3_placeholder(self) -> None:
        """Placeholder - Phase 3 removed"""
        return
            
        try:
            # Check if any Phase 3 features are enabled
            needs_phase3 = any([
                getattr(self.settings, 'enable_domain_classification', False),
                getattr(self.settings, 'enable_quality_scoring', False),
                getattr(self.settings, 'enable_cps_optimization', False),
                getattr(self.settings, 'enable_vocabulary_enforcement', False),
                getattr(self.settings, 'enable_adaptive_processing', False)
            ])
            
            if not needs_phase3:
                return
                
            # Initialize domain classifier
            if getattr(self.settings, 'enable_domain_classification', False):
                self.phase3_components['domain_classifier'] = DomainClassifier()
                logger.info("Domain classifier initialized")
                
            # Initialize quality scorer
            if getattr(self.settings, 'enable_quality_scoring', False):
                self.phase3_components['quality_scorer'] = QualityScorer()
                logger.info("Quality scorer initialized")
                
            # Initialize CPS optimizer
            if getattr(self.settings, 'enable_cps_optimization', False):
                self.phase3_components['cps_optimizer'] = CPSOptimizer(
                    preserve_formal=False  # Could be configurable
                )
                logger.info("CPS optimizer initialized")
                
            # Initialize vocabulary manager
            if getattr(self.settings, 'enable_vocabulary_enforcement', False):
                self.phase3_components['vocabulary_manager'] = VocabularyManager()
                logger.info("Vocabulary manager initialized")
                
            # Initialize adaptive processor
            if getattr(self.settings, 'enable_adaptive_processing', False):
                self.phase3_components['adaptive_processor'] = AdaptiveProcessor()
                logger.info("Adaptive processor initialized")
                
            self._phase3_initialized = True
            logger.info(f"Phase 3 components initialized: {list(self.phase3_components.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Phase 3 components: {e}")
            self._phase3_initialized = False
                
    async def clean(self, 
                   document: SubtitleDocument,
                   context_sources: Optional[List[Dict]] = None) -> SubtitleDocument:
        """Clean subtitle document through processing pipeline with enhanced features"""
        start_time = time.time()
        
        # Initialize models if needed (Phase 2/3 removed)
        await self.initialize_ml()
        await self.initialize_onnx_punctuation()
        
        # Create a deep copy to avoid modifying the original
        doc = deepcopy(document)
        self.modifications = []
        
        # Reset change tracker for this document
        self.change_tracker = ChangeTracker()
        self.change_tracker.set_original_document([
            {'idx': s.idx, 'text': s.text, 'start_ms': s.start_ms, 'end_ms': s.end_ms}
            for s in doc.segments
        ])
        
        # Extract context if provided
        context_lexicon = {}
        if context_sources:
            try:
                context_result = await self.context_extractor.extract_from_sources(
                    context_sources
                )
                context_lexicon = context_result.get('lexicon', {})
                logger.info(f"Extracted {len(context_lexicon)} entities from context")
            except Exception as e:
                logger.warning(f"Context extraction failed: {e}")
        
        # Pipeline steps with change tracking
        
        # 1. Merge segments if needed
        if self.settings.merge_mode != MergeMode.OFF:
            doc = self._merge_segments(doc)
        
        # 2. Apply punctuation restoration if enabled
        if self._onnx_initialized and self.onnx_punct_model:
            doc = await self._apply_punctuation_restoration(doc)
        
        # 3. Apply entity stabilization if enabled
        if getattr(self.settings, 'enable_entity_stabilization', False):
            doc = self._apply_entity_stabilization(doc, context_lexicon)
        
        # 4. Line wrapping
        doc = self._wrap_lines(doc)
        
        # 5. Apply ML enhancement or basic normalization
        if self._ml_initialized:
            doc = await self._apply_ml_enhancement(doc)
        else:
            doc = self._normalize_text(doc)
            
            if self.settings.filler_mode != FillerMode.KEEP:
                doc = self._remove_fillers(doc)
        
        # 6. Fix timing issues
        doc = self._fix_timing_issues(doc)
        
        # 7. Apply final guardrails
        doc = self._apply_guardrails(doc)
        
        # 8. Apply context lexicon if available
        if context_lexicon:
            doc = self._apply_context_lexicon(doc, context_lexicon)
        
        # Phase 2 and 3 processing removed - not part of core requirements
        
        # Finalize change tracking
        self.change_tracker.set_final_document([
            {'idx': s.idx, 'text': s.text, 'start_ms': s.start_ms, 'end_ms': s.end_ms}
            for s in doc.segments
        ])
        self.change_tracker._processing_time_ms = int((time.time() - start_time) * 1000)
        
        return doc
    
    async def _apply_ml_enhancement(self, doc: SubtitleDocument) -> SubtitleDocument:
        """Apply ML models to enhance subtitle quality"""
        ml_stats = {
            "segments_enhanced": 0,
            "segments_rejected": 0,
            "models_used": list(self.ml_models.keys())
        }
        
        for segment in doc.segments:
            original_text = segment.text
            current_text = original_text
            
            # Apply punctuation correction
            if 'punctuation' in self.ml_models:
                result = await self.ml_models['punctuation'].predict(current_text)
                if self.gatekeeper and self.gatekeeper.validate(result):
                    current_text = result.predicted_text
                    ml_stats["segments_enhanced"] += 1
                else:
                    ml_stats["segments_rejected"] += 1
            
            # Apply grammar correction
            if 'grammar' in self.ml_models:
                result = await self.ml_models['grammar'].predict(current_text)
                if self.gatekeeper and self.gatekeeper.validate(result):
                    current_text = result.predicted_text
            
            # Apply contextual filler removal
            if 'filler' in self.ml_models:
                result = await self.ml_models['filler'].predict(current_text)
                if self.gatekeeper and self.gatekeeper.validate(result):
                    current_text = result.predicted_text
            
            # Score improvement if scorer available
            if 'scorer' in self.ml_models and current_text != original_text:
                original_score = await self.ml_models['scorer'].score(original_text)
                new_score = await self.ml_models['scorer'].score(current_text)
                
                # Only accept if perplexity improved (lower is better)
                if new_score < original_score:
                    segment.text = current_text
                    self.modifications.append({
                        "type": "ml_enhancement",
                        "segment": segment.idx,
                        "perplexity_improvement": original_score - new_score
                    })
                else:
                    segment.text = original_text
            else:
                segment.text = current_text
        
        # Add ML stats to modifications
        self.modifications.append({
            "type": "ml_summary",
            "stats": ml_stats
        })
        
        return doc
    
    def _merge_segments(self, doc: SubtitleDocument) -> SubtitleDocument:
        """Merge short segments intelligently"""
        if not doc.segments:
            return doc
        
        merged_segments = []
        i = 0
        
        while i < len(doc.segments):
            current = doc.segments[i]
            
            # Try to merge with next segment
            if i + 1 < len(doc.segments):
                next_seg = doc.segments[i + 1]
                
                if self._should_merge(current, next_seg):
                    # Merge segments
                    merged = Segment(
                        idx=current.idx,
                        start_ms=current.start_ms,
                        end_ms=next_seg.end_ms,
                        text=self._merge_text(current.text, next_seg.text)
                    )
                    
                    # Check if merged segment is valid
                    if merged.cps <= self.settings.max_cps:
                        merged_segments.append(merged)
                        self.modifications.append({
                            "type": "merge",
                            "segments": [current.idx, next_seg.idx],
                            "reason": "short_duration"
                        })
                        i += 2  # Skip next segment
                        continue
            
            merged_segments.append(current)
            i += 1
        
        # Re-index segments
        for idx, seg in enumerate(merged_segments, 1):
            seg.idx = idx
        
        return SubtitleDocument(segments=merged_segments)
    
    def _should_merge(self, seg1: Segment, seg2: Segment) -> bool:
        """Determine if two segments should be merged"""
        # Check mode
        if self.settings.merge_mode == MergeMode.OFF:
            return False
        
        # Check gap between segments
        gap_ms = seg2.start_ms - seg1.end_ms
        if gap_ms > 1000:  # More than 1 second gap
            return False
        
        # Check combined duration
        combined_duration = seg2.end_ms - seg1.start_ms
        if combined_duration > self.settings.max_duration_ms:
            return False
        
        # Mode-specific checks
        if self.settings.merge_mode == MergeMode.CONSERVATIVE:
            # Only merge very short segments
            return seg1.duration_ms < 1000 or seg2.duration_ms < 1000
        
        elif self.settings.merge_mode == MergeMode.SMART:
            # Merge if either is short or both are below target
            short_threshold = self.settings.min_duration_ms * 0.8
            return (
                seg1.duration_ms < short_threshold or 
                seg2.duration_ms < short_threshold or
                (seg1.duration_ms < self.settings.min_duration_ms and 
                 seg2.duration_ms < self.settings.min_duration_ms)
            )
        
        elif self.settings.merge_mode == MergeMode.AGGRESSIVE:
            # Merge if both are below target duration
            return (
                seg1.duration_ms < self.settings.min_duration_ms and
                seg2.duration_ms < self.settings.min_duration_ms
            )
        
        return False
    
    def _merge_text(self, text1: str, text2: str) -> str:
        """Merge text from two segments"""
        # Clean up texts
        text1 = text1.strip()
        text2 = text2.strip()
        
        # Check if text1 ends with punctuation
        if text1 and text1[-1] in '.!?':
            return f"{text1} {text2}"
        else:
            # Might be continuation
            return f"{text1} {text2}"
    
    def _wrap_lines(self, doc: SubtitleDocument) -> SubtitleDocument:
        """Wrap long lines in segments"""
        for segment in doc.segments:
            lines = segment.text.split('\n')
            wrapped_lines = []
            
            for line in lines:
                if len(line) <= self.settings.line_wrap:
                    wrapped_lines.append(line)
                else:
                    # Wrap long line
                    wrapped = self._wrap_single_line(line)
                    wrapped_lines.extend(wrapped)
            
            # Limit to max lines
            if len(wrapped_lines) > self.settings.max_lines:
                # Try to combine or truncate
                wrapped_lines = wrapped_lines[:self.settings.max_lines]
                self.modifications.append({
                    "type": "truncate_lines",
                    "segment": segment.idx,
                    "original_lines": len(lines),
                    "new_lines": len(wrapped_lines)
                })
            
            segment.text = '\n'.join(wrapped_lines)
        
        return doc
    
    def _wrap_single_line(self, text: str) -> List[str]:
        """Wrap a single long line into multiple lines"""
        if len(text) <= self.settings.line_wrap:
            return [text]
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            
            # Check if adding word would exceed limit
            if current_length + word_length + len(current_line) > self.settings.line_wrap:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = word_length
                else:
                    # Single word is too long, force break
                    lines.append(word)
            else:
                current_line.append(word)
                current_length += word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines[:self.settings.max_lines]
    
    def _normalize_text(self, doc: SubtitleDocument) -> SubtitleDocument:
        """Normalize punctuation and formatting"""
        for segment in doc.segments:
            text = segment.text
            
            # Remove HTML/formatting tags if not preserving
            if not self.settings.preserve_formatting:
                text = re.sub(TAG_PATTERN, '', text)
            
            # Normalize punctuation
            for old, new in PUNCTUATION_RULES.items():
                text = text.replace(old, new)
            
            # Normalize whitespace
            text = re.sub(MULTIPLE_SPACES_PATTERN, ' ', text)
            text = text.strip()
            
            # Fix common issues
            text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
            text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation
            
            segment.text = text
        
        return doc
    
    def _remove_fillers(self, doc: SubtitleDocument) -> SubtitleDocument:
        """Remove filler words/phrases"""
        language = Language(self.settings.language)
        fillers = FILLERS.get(language, set())
        
        # Add custom fillers
        if hasattr(self.settings, 'custom_fillers'):
            fillers = fillers.union(set(self.settings.custom_fillers))
        
        if not fillers:
            return doc
        
        for segment in doc.segments:
            original = segment.text
            text = segment.text
            
            # Remove fillers based on mode
            if self.settings.filler_mode == FillerMode.REMOVE:
                text = self._remove_all_fillers(text, fillers)
            elif self.settings.filler_mode == FillerMode.SMART:
                text = self._remove_smart_fillers(text, fillers)
            
            if text != original:
                self.modifications.append({
                    "type": "remove_filler",
                    "segment": segment.idx,
                    "removed": self._get_removed_fillers(original, text, fillers)
                })
            
            segment.text = text
        
        return doc
    
    def _remove_all_fillers(self, text: str, fillers: Set[str]) -> str:
        """Remove all filler words"""
        words = text.split()
        cleaned = []
        
        for word in words:
            # Check if word (without punctuation) is a filler
            clean_word = re.sub(r'[^\w\s]', '', word.lower())
            if clean_word not in fillers:
                cleaned.append(word)
        
        return ' '.join(cleaned)
    
    def _remove_smart_fillers(self, text: str, fillers: Set[str]) -> str:
        """Remove fillers intelligently (keep some for natural flow)"""
        # For smart mode, only remove excessive fillers
        # Keep first occurrence in a segment
        words = text.split()
        cleaned = []
        seen_fillers = set()
        
        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word.lower())
            
            if clean_word in fillers:
                if clean_word not in seen_fillers:
                    # Keep first occurrence
                    cleaned.append(word)
                    seen_fillers.add(clean_word)
                # Skip subsequent occurrences
            else:
                cleaned.append(word)
        
        return ' '.join(cleaned)
    
    def _get_removed_fillers(self, original: str, cleaned: str, fillers: Set[str]) -> List[str]:
        """Get list of removed filler words"""
        original_words = set(re.sub(r'[^\w\s]', '', w.lower()) for w in original.split())
        cleaned_words = set(re.sub(r'[^\w\s]', '', w.lower()) for w in cleaned.split())
        removed = original_words - cleaned_words
        return list(removed.intersection(fillers))
    
    def _fix_timing_issues(self, doc: SubtitleDocument) -> SubtitleDocument:
        """Fix overlaps and timing violations"""
        segments = doc.segments
        
        # Fix overlaps
        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]
            
            if current.end_ms > next_seg.start_ms:
                # Overlap detected
                overlap = current.end_ms - next_seg.start_ms
                
                # Simple fix: adjust end time of current
                current.end_ms = next_seg.start_ms - 1
                
                self.modifications.append({
                    "type": "fix_overlap",
                    "segments": [current.idx, next_seg.idx],
                    "overlap_ms": overlap
                })
        
        # Fix duration violations
        for segment in segments:
            if segment.duration_ms < self.settings.min_duration_ms:
                # Try to extend if possible
                new_end = segment.start_ms + self.settings.min_duration_ms
                
                # Check if it would overlap with next
                idx = segments.index(segment)
                if idx + 1 < len(segments):
                    next_seg = segments[idx + 1]
                    if new_end >= next_seg.start_ms:
                        new_end = next_seg.start_ms - 1
                
                if new_end > segment.end_ms:
                    segment.end_ms = new_end
                    self.modifications.append({
                        "type": "extend_duration",
                        "segment": segment.idx,
                        "new_duration_ms": segment.duration_ms
                    })
        
        return doc
    
    def _apply_guardrails(self, doc: SubtitleDocument) -> SubtitleDocument:
        """Apply final safety checks and limits"""
        for segment in doc.segments:
            # Enforce maximum CPS by extending duration if needed
            if segment.cps > self.settings.max_cps:
                # Calculate required duration for target CPS
                required_duration = len(segment.text) / self.settings.max_cps * 1000
                
                if required_duration > segment.duration_ms:
                    # Try to extend end time
                    new_end = segment.start_ms + int(required_duration)
                    
                    # Check constraints
                    if new_end - segment.start_ms <= self.settings.max_duration_ms:
                        segment.end_ms = new_end
                        self.modifications.append({
                            "type": "adjust_cps",
                            "segment": segment.idx,
                            "original_cps": segment.cps,
                            "new_duration_ms": segment.duration_ms
                        })
        
        return doc
    
    async def _apply_punctuation_restoration(self, doc: SubtitleDocument) -> SubtitleDocument:
        """Apply ONNX punctuation restoration to segments"""
        for i, segment in enumerate(doc.segments):
            original = segment.text
            
            # Get context from surrounding segments
            context = []
            if i > 0:
                context.append(doc.segments[i-1].text)
                
            # Restore punctuation
            restored = self.onnx_punct_model.restore(original, context)
            
            if restored != original:
                segment.text = restored
                confidence = self.onnx_punct_model.get_confidence(original, restored)
                
                # Track change
                self.change_tracker.track_text_change(
                    segment_idx=segment.idx,
                    old_text=original,
                    new_text=restored,
                    change_type=ChangeType.PUNCTUATION_RESTORED,
                    reason="Applied punctuation and truecasing restoration",
                    source=ChangeSource.ONNX_PUNCT,
                    confidence=confidence
                )
                
        return doc
        
    def _apply_entity_stabilization(self, doc: SubtitleDocument, context_lexicon: Dict) -> SubtitleDocument:
        """Apply entity stabilization to fix name variants"""
        
        # Convert to format expected by stabilizer
        segments = [
            {'text': seg.text, 'idx': seg.idx}
            for seg in doc.segments
        ]
        
        # Run stabilization
        stabilized_segments, report = self.entity_stabilizer.stabilize_document(segments)
        
        # Apply changes
        for seg, stab_seg in zip(doc.segments, stabilized_segments):
            if seg.text != stab_seg['text']:
                old_text = seg.text
                seg.text = stab_seg['text']
                
                # Track change
                self.change_tracker.track_text_change(
                    segment_idx=seg.idx,
                    old_text=old_text,
                    new_text=seg.text,
                    change_type=ChangeType.ENTITY_STABILIZED,
                    reason="Stabilized entity variants for consistency",
                    source=ChangeSource.PHONETIC_MATCH,
                    confidence=0.8,
                    metadata={"clusters": len(report.get("clusters", []))}
                )
                
        return doc
        
    def _apply_context_lexicon(self, doc: SubtitleDocument, lexicon: Dict) -> SubtitleDocument:
        """Apply entities from context sources"""
        
        for segment in doc.segments:
            original = segment.text
            modified = original
            
            # Apply high-confidence replacements from lexicon
            for entity, info in lexicon.items():
                if info['confidence'] >= 0.8 and not info.get('protected'):
                    # Case-insensitive replacement
                    pattern = re.compile(re.escape(entity), re.IGNORECASE)
                    if pattern.search(modified):
                        modified = pattern.sub(entity, modified)
                        
            if modified != original:
                segment.text = modified
                
                # Track change
                self.change_tracker.track_text_change(
                    segment_idx=segment.idx,
                    old_text=original,
                    new_text=modified,
                    change_type=ChangeType.GLOSSARY_APPLIED,
                    reason="Applied entity from user-provided context",
                    source=ChangeSource.USER_CONTEXT,
                    confidence=0.9
                )
                
        return doc
    
    # Phase 2 processing removed
    async def _phase2_processing_removed(self, doc: SubtitleDocument) -> None:
        """Apply Phase 2 contextual understanding processing"""
        # Convert segments to format needed by Phase 2
        segments_data = [
            {
                'idx': seg.idx,
                'text': seg.text,
                'start_ms': seg.start_ms,
                'end_ms': seg.end_ms
            }
            for seg in doc.segments
        ]
        
        # Topic segmentation
        if 'topic_segmenter' in self.phase2_components:
            try:
                topics = self.phase2_components['topic_segmenter'].segment_document(segments_data)
                self.phase2_results['topics'] = {
                    'num_topics': len(topics),
                    'topics': [
                        {
                            'id': t.id,
                            'start': t.start_idx,
                            'end': t.end_idx,
                            'coherence': t.coherence_score,
                            'keywords': t.keywords[:5]
                        }
                        for t in topics
                    ]
                }
                logger.info(f"Topic segmentation found {len(topics)} topics")
            except Exception as e:
                logger.warning(f"Topic segmentation failed: {e}")
        
        # Speaker tracking
        if 'speaker_tracker' in self.phase2_components:
            try:
                speaker_map = self.phase2_components['speaker_tracker'].track_speakers(segments_data)
                # Merge similar speakers after initial tracking
                self.phase2_components['speaker_tracker'].merge_similar_speakers(threshold=0.7)
                speaker_report = self.phase2_components['speaker_tracker'].get_speaker_report()
                self.phase2_results['speakers'] = speaker_report
                logger.info(f"Speaker tracking found {speaker_report.get('num_speakers', 0)} speakers")
            except Exception as e:
                logger.warning(f"Speaker tracking failed: {e}")
        
        # Coreference resolution
        if 'coreference_resolver' in self.phase2_components:
            try:
                resolutions = self.phase2_components['coreference_resolver'].resolve_document(segments_data)
                coref_report = self.phase2_components['coreference_resolver'].get_coreference_report()
                
                # Apply resolutions to document if configured
                if getattr(self.settings, 'apply_coreference_resolutions', False):
                    for seg_idx, resolution in resolutions.items():
                        if resolution['resolved_text'] and resolution['mappings']:
                            # Find segment
                            segment = next((s for s in doc.segments if s.idx == seg_idx), None)
                            if segment:
                                segment.text = resolution['resolved_text']
                                # Track change
                                self.change_tracker.track_text_change(
                                    segment_idx=seg_idx,
                                    old_text=segments_data[seg_idx-1]['text'],
                                    new_text=resolution['resolved_text'],
                                    change_type=ChangeType.OTHER,
                                    reason="Coreference resolution applied",
                                    source=ChangeSource.OTHER,
                                    metadata={'mappings': resolution['mappings']}
                                )
                
                self.phase2_results['coreferences'] = coref_report
                logger.info(f"Coreference resolution found {coref_report.get('num_chains', 0)} chains")
            except Exception as e:
                logger.warning(f"Coreference resolution failed: {e}")
        
    def get_modifications_report(self) -> Dict[str, Any]:
        """Get comprehensive report including change tracking"""
        # Use new change tracker report if available
        if self.change_tracker and self.change_tracker.changes:
            return self.change_tracker.generate_report()
            
        # Fallback to old report format
        """Get report of all modifications made"""
        return {
            "total_modifications": len(self.modifications),
            "modifications": self.modifications,
            "summary": self._summarize_modifications()
        }
    
    def _summarize_modifications(self) -> Dict[str, int]:
        """Summarize modifications by type"""
        summary = {}
        for mod in self.modifications:
            mod_type = mod["type"]
            summary[mod_type] = summary.get(mod_type, 0) + 1
        return summary
    
    # Phase 3 processing removed
    async def _phase3_processing_removed(self, doc: SubtitleDocument) -> None:
        """Apply Phase 3 advanced processing (domain-specific, quality optimization)"""
        try:
            logger.info("Applying Phase 3 advanced processing")
            
            # Use adaptive processor if enabled
            if (getattr(self.settings, 'enable_adaptive_processing', False) and
                'adaptive_processor' in self.phase3_components):
                
                # Adaptive processing handles everything
                result = await self.phase3_components['adaptive_processor'].process(doc)
                
                self.phase3_results = {
                    'strategy_used': result.strategy_used.value,
                    'domain_detected': result.domain_detected,
                    'average_quality_score': result.average_quality_score,
                    'segments_modified': result.segments_modified,
                    'quality_improvement': result.quality_improvements.get('improvement', 0),
                    'warnings': result.warnings
                }
                
                # Track changes
                for mod in result.modifications:
                    self.change_tracker.add_change(
                        segment_idx=mod['segment'],
                        change_type=ChangeType.STYLE,  # Or appropriate type
                        original="",  # Would need original text
                        modified="",  # Would need modified text
                        source=ChangeSource.ML,
                        confidence=0.9,
                        metadata={'phase3': mod['modifications']}
                    )
                    
            else:
                # Apply individual Phase 3 components
                phase3_stats = {}
                
                # 1. Domain classification
                if (getattr(self.settings, 'enable_domain_classification', False) and
                    'domain_classifier' in self.phase3_components):
                    
                    segments_data = [{'text': s.text} for s in doc.segments[:20]]
                    domain_profile = self.phase3_components['domain_classifier'].classify(segments_data)
                    phase3_stats['domain'] = domain_profile.domain
                    phase3_stats['domain_confidence'] = domain_profile.confidence
                    logger.info(f"Domain detected: {domain_profile.domain} (confidence: {domain_profile.confidence:.2f})")
                    
                # 2. Quality scoring
                quality_scores = []
                if (getattr(self.settings, 'enable_quality_scoring', False) and
                    'quality_scorer' in self.phase3_components):
                    
                    for segment in doc.segments:
                        seg_dict = {'text': segment.text, 'start': segment.start_time, 'end': segment.end_time}
                        report = self.phase3_components['quality_scorer'].score(seg_dict)
                        quality_scores.append(report.overall_score)
                        
                        # Store quality issues
                        if report.issues:
                            self.change_tracker.add_change(
                                segment_idx=segment.index,
                                change_type=ChangeType.STYLE,
                                original=segment.text,
                                modified=segment.text,
                                source=ChangeSource.ML,
                                confidence=report.confidence,
                                metadata={'quality_issues': report.issues}
                            )
                    
                    if quality_scores:
                        phase3_stats['average_quality'] = sum(quality_scores) / len(quality_scores)
                        phase3_stats['min_quality'] = min(quality_scores)
                        phase3_stats['max_quality'] = max(quality_scores)
                        
                # 3. CPS optimization
                if (getattr(self.settings, 'enable_cps_optimization', False) and
                    'cps_optimizer' in self.phase3_components):
                    
                    optimized_count = 0
                    for segment in doc.segments:
                        seg_dict = {
                            'text': segment.text,
                            'start': segment.start_time,
                            'end': segment.end_time
                        }
                        
                        result = self.phase3_components['cps_optimizer'].optimize(seg_dict)
                        
                        if result.optimized_text != result.original_text:
                            segment.text = result.optimized_text
                            optimized_count += 1
                            
                            self.change_tracker.add_change(
                                segment_idx=segment.index,
                                change_type=ChangeType.STYLE,
                                original=result.original_text,
                                modified=result.optimized_text,
                                source=ChangeSource.ML,
                                confidence=result.confidence,
                                metadata={
                                    'cps_before': result.original_cps,
                                    'cps_after': result.optimized_cps
                                }
                            )
                    
                    phase3_stats['cps_optimized'] = optimized_count
                    
                # 4. Vocabulary enforcement
                if (getattr(self.settings, 'enable_vocabulary_enforcement', False) and
                    'vocabulary_manager' in self.phase3_components):
                    
                    # Get domain from classifier or use general
                    domain = phase3_stats.get('domain', 'general') if phase3_stats.get('domain_confidence', 0) > 0.7 else None
                    custom_glossary = getattr(self.settings, 'custom_glossary', None)
                    
                    vocab_count = 0
                    for segment in doc.segments:
                        enforced_text, matches = self.phase3_components['vocabulary_manager'].enforce_glossary(
                            segment.text,
                            domain=domain,
                            custom_glossary=custom_glossary
                        )
                        
                        if enforced_text != segment.text:
                            original_text = segment.text
                            segment.text = enforced_text
                            vocab_count += 1
                            
                            self.change_tracker.add_change(
                                segment_idx=segment.index,
                                change_type=ChangeType.TERMINOLOGY,
                                original=original_text,
                                modified=enforced_text,
                                source=ChangeSource.ML,
                                confidence=0.95,
                                metadata={'glossary_matches': len(matches)}
                            )
                    
                    phase3_stats['vocabulary_enforced'] = vocab_count
                    
                self.phase3_results = phase3_stats
                
            logger.info(f"Phase 3 processing complete: {self.phase3_results}")
            
        except Exception as e:
            logger.error(f"Error in Phase 3 processing: {e}", exc_info=True)
            self.phase3_results = {'error': str(e)}