"""Document Normalizer - Statistical Pattern-Based Entity Normalization

This module provides intelligent normalization of ambiguous words and entities
by analyzing document-wide patterns and frequencies.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Import the contextual correction engine
try:
    from app.services.contextual_correction_engine import (
        integrate_contextual_engine,
        CorrectionMode
    )
    CONTEXTUAL_ENGINE_AVAILABLE = True
except ImportError:
    CONTEXTUAL_ENGINE_AVAILABLE = False
    logger.warning("ContextualCorrectionEngine not available")


@dataclass
class EntityVariation:
    """Represents a variation of an entity found in the document"""
    text: str
    frequency: int
    positions: List[int]  # Character positions in document
    confidence: float
    is_likely_correct: bool


@dataclass
class NormalizationCandidate:
    """A potential normalization from one form to another"""
    original: str
    normalized: str
    confidence: float
    reason: str
    frequency_ratio: float  # normalized_freq / original_freq


class DocumentNormalizer:
    """
    Normalizes entities within a document using statistical patterns.
    
    Key features:
    - Identifies frequently occurring entities as likely correct
    - Detects rare variations as potential misrecognitions
    - Handles partial matches (surnames, truncations)
    - Uses context and co-occurrence for validation
    """
    
    def __init__(self):
        # Minimum frequency to consider an entity as "established"
        self.min_frequency_for_correct = 2
        
        # Maximum frequency for a potential error
        self.max_frequency_for_error = 2
        
        # Confidence thresholds
        self.min_confidence_for_normalization = 0.7
        
        # Pattern for extracting capitalized words/phrases
        self.entity_pattern = re.compile(
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        )
        
        # Common words to ignore
        self.stop_words = {
            'The', 'This', 'That', 'These', 'Those', 'What', 'When', 
            'Where', 'Who', 'Why', 'How', 'Which', 'While', 'After',
            'Before', 'During', 'Since', 'Until', 'Although', 'Because'
        }
    
    def normalize_document(
        self,
        document_text: str,
        context_entities: Optional[Dict[str, any]] = None,
        use_contextual_engine: bool = True,
        correction_mode: str = "balanced"
    ) -> Dict[str, str]:
        """
        Main entry point for document normalization.
        
        Args:
            document_text: The text to normalize
            context_entities: Optional context entities for matching
            use_contextual_engine: Whether to use ML-enhanced correction engine
            correction_mode: Mode for contextual engine ("legacy", "conservative", "balanced", "aggressive")
        
        Returns:
            Dictionary of original -> normalized mappings
        """
        logger.info(f"Document normalizer called with text length {len(document_text)}")
        logger.info(f"Context entities provided: {len(context_entities) if context_entities else 0}")
        if context_entities:
            # Log what entities we received
            entity_names = [e.text if hasattr(e, 'text') else str(e) for e in list(context_entities.values())[:10]]
            logger.info(f"Context entity samples: {entity_names}")
        
        # Step 1: Extract all entities and their frequencies
        entity_variations = self._extract_entity_variations(document_text)
        logger.info(f"Found {len(entity_variations)} entity variations")
        
        # Step 2: Identify likely correct forms
        correct_entities = self._identify_correct_entities(entity_variations)
        logger.info(f"Identified {len(correct_entities)} correct entities: {list(correct_entities)[:5]}")
        
        # Step 3: Find normalization candidates
        candidates = self._find_normalization_candidates(
            entity_variations, 
            correct_entities,
            context_entities
        )
        logger.info(f"Found {len(candidates)} normalization candidates")
        
        # Step 4: Build final normalization map
        normalizations = self._build_normalization_map(candidates)
        
        # Step 5: Apply contextual correction engine if available and enabled
        if use_contextual_engine and CONTEXTUAL_ENGINE_AVAILABLE and normalizations:
            logger.info(f"Applying contextual correction engine in {correction_mode} mode")
            try:
                normalizations = integrate_contextual_engine(
                    self,
                    normalizations,
                    document_text,
                    context_entities if context_entities else {},
                    mode=correction_mode
                )
                logger.info(f"After contextual filtering: {len(normalizations)} corrections remain")
            except Exception as e:
                logger.error(f"Contextual engine failed: {e}, using original normalizations")
        
        logger.info(f"Document normalization found {len(normalizations)} corrections")
        for orig, norm in list(normalizations.items())[:5]:
            logger.info(f"  Normalize: '{orig}' -> '{norm}'")
        
        return normalizations
    
    def _extract_entity_variations(self, text: str) -> Dict[str, EntityVariation]:
        """Extract all entity-like patterns and their variations"""
        variations = {}
        
        # Find all potential entities
        matches = self.entity_pattern.finditer(text)
        entity_positions = defaultdict(list)
        
        for match in matches:
            entity_text = match.group()
            
            # Skip common words
            if entity_text in self.stop_words:
                continue
            
            # Track positions for context analysis
            entity_positions[entity_text].append(match.start())
        
        # Create variation objects
        for entity_text, positions in entity_positions.items():
            frequency = len(positions)
            
            # Calculate base confidence based on frequency
            if frequency >= 3:
                confidence = 0.9
            elif frequency >= 2:
                confidence = 0.7
            else:
                confidence = 0.4
            
            variations[entity_text] = EntityVariation(
                text=entity_text,
                frequency=frequency,
                positions=positions,
                confidence=confidence,
                is_likely_correct=frequency >= self.min_frequency_for_correct
            )
        
        # Also extract partial names (last names only)
        self._extract_partial_names(text, variations)
        
        return variations
    
    def _extract_partial_names(
        self, 
        text: str, 
        variations: Dict[str, EntityVariation]
    ):
        """Extract last names from full names for matching"""
        # For each multi-word entity, also consider the last word alone
        for entity_text in list(variations.keys()):
            words = entity_text.split()
            if len(words) >= 2:
                last_word = words[-1]
                
                # Only add if it's substantial enough
                if len(last_word) >= 4 and last_word not in variations:
                    # Count occurrences of just the last name
                    pattern = r'\b' + re.escape(last_word) + r'\b'
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    positions = [m.start() for m in matches]
                    
                    if positions:
                        variations[last_word] = EntityVariation(
                            text=last_word,
                            frequency=len(positions),
                            positions=positions,
                            confidence=0.6,  # Lower confidence for partial
                            is_likely_correct=False  # Will be determined by context
                        )
    
    def _identify_correct_entities(
        self,
        variations: Dict[str, EntityVariation]
    ) -> Set[str]:
        """Identify entities that are likely spelled correctly"""
        correct = set()
        
        for entity_text, variation in variations.items():
            # High frequency suggests correct spelling
            if variation.frequency >= self.min_frequency_for_correct:
                correct.add(entity_text)
                
                # Also add the full form if this is a last name
                # that appears as part of a full name
                for other_text, other_var in variations.items():
                    if other_text != entity_text and entity_text in other_text:
                        if other_var.frequency >= 2:
                            correct.add(other_text)
        
        return correct
    
    def _find_normalization_candidates(
        self,
        variations: Dict[str, EntityVariation],
        correct_entities: Set[str],
        context_entities: Optional[Dict] = None
    ) -> List[NormalizationCandidate]:
        """Find potential normalizations based on patterns"""
        candidates = []
        
        # Build a set of authoritative entities from context if provided
        context_entity_texts = set()
        if context_entities:
            for entity in context_entities.values():
                # Handle both ExtractedEntity objects and raw strings
                if hasattr(entity, 'text'):
                    context_entity_texts.add(entity.text)
                else:
                    context_entity_texts.add(str(entity))
            
            # Add context entities to correct entities for matching
            correct_entities = correct_entities.union(context_entity_texts)
            logger.info(f"Added {len(context_entity_texts)} context entities to correct set")
        
        # Check each rare entity against both frequent ones AND context entities
        for entity_text, variation in variations.items():
            # Skip if already identified as correct
            if entity_text in correct_entities:
                continue
            
            # Skip if too frequent to be an error
            if variation.frequency > self.max_frequency_for_error:
                continue
            
            # Find potential corrections from both document patterns and context
            for correct_text in correct_entities:
                # Ensure we have the variation data for frequency ratio
                # Context entities might not be in variations dict
                if correct_text in variations:
                    freq_ratio = variations[correct_text].frequency / variation.frequency
                else:
                    # Context entities get high confidence if they match
                    freq_ratio = 10.0  # High ratio for context entities
                
                confidence, reason = self._calculate_match_confidence(
                    entity_text, 
                    correct_text,
                    variations
                )
                
                # Boost confidence if it's from context
                if correct_text in context_entity_texts and confidence > 0.5:
                    confidence = min(confidence + 0.2, 0.95)
                    reason = f"context_{reason}"
                
                if confidence >= self.min_confidence_for_normalization:
                    candidates.append(NormalizationCandidate(
                        original=entity_text,
                        normalized=correct_text,
                        confidence=confidence,
                        reason=reason,
                        frequency_ratio=freq_ratio
                    ))
        
        # Sort by confidence
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        return candidates
    
    def _calculate_match_confidence(
        self,
        candidate: str,
        target: str,
        variations: Dict[str, EntityVariation]
    ) -> Tuple[float, str]:
        """Calculate confidence that candidate should be normalized to target"""
        candidate_lower = candidate.lower()
        target_lower = target.lower()
        
        # Check for suffix match (like "Mecano" -> "Upamecano")
        if target_lower.endswith(candidate_lower):
            # Strong signal if target is frequent and candidate is rare
            # Handle case where target is from context (not in variations)
            if target in variations:
                freq_ratio = variations[target].frequency / variations[candidate].frequency
            else:
                # Context entity - assume high authority
                freq_ratio = 5.0
            if freq_ratio >= 3:
                return 0.9, "suffix_match_high_frequency"
            elif freq_ratio >= 2:
                return 0.8, "suffix_match_medium_frequency"
            else:
                return 0.7, "suffix_match_low_frequency"
        
        # Check for prefix match (like "Man" -> "Manchester")
        if target_lower.startswith(candidate_lower):
            if target in variations:
                freq_ratio = variations[target].frequency / variations[candidate].frequency
            else:
                freq_ratio = 5.0
            if freq_ratio >= 3:
                return 0.85, "prefix_match_high_frequency"
            else:
                return 0.75, "prefix_match"
        
        # Check if candidate is last word of target (surname matching)
        target_words = target.split()
        if len(target_words) >= 2 and candidate == target_words[-1]:
            return 0.85, "surname_match"
        
        # Fuzzy matching for close spellings
        similarity = SequenceMatcher(None, candidate_lower, target_lower).ratio()
        
        # Adjust based on frequency
        if target in variations:
            freq_ratio = variations[target].frequency / variations[candidate].frequency
        else:
            freq_ratio = 5.0
        
        if similarity > 0.8:
            if freq_ratio >= 3:
                return min(0.95, similarity + 0.1), "fuzzy_match_high_frequency"
            else:
                return similarity, "fuzzy_match"
        
        # Check for partial word matches
        if candidate_lower in target_lower or target_lower in candidate_lower:
            if freq_ratio >= 2:
                return 0.75, "partial_match"
        
        return 0.0, "no_match"
    
    def _build_normalization_map(
        self,
        candidates: List[NormalizationCandidate]
    ) -> Dict[str, str]:
        """Build final normalization map, resolving conflicts"""
        normalizations = {}
        
        # Track what's already been normalized to prevent chains
        normalized_forms = set()
        
        for candidate in candidates:
            # Skip if already normalized
            if candidate.original in normalizations:
                continue
            
            # Skip if would create a chain (A->B->C)
            if candidate.original in normalized_forms:
                continue
            
            # Add the normalization
            normalizations[candidate.original] = candidate.normalized
            normalized_forms.add(candidate.normalized)
            
            logger.debug(
                f"Normalization ({candidate.reason}, conf={candidate.confidence:.2f}): "
                f"'{candidate.original}' -> '{candidate.normalized}' "
                f"(freq ratio: {candidate.frequency_ratio:.1f})"
            )
        
        return normalizations


class EnhancedDocumentNormalizer(DocumentNormalizer):
    """Enhanced version with additional intelligent features"""
    
    def __init__(self):
        super().__init__()
        
        # Known problematic patterns in ASR
        self.common_misrecognitions = {
            'may united': 'man united',
            'manchester': 'manchester united',
            'tottenham': 'tottenham hotspur',
            'bayern': 'bayern munich',
        }
    
    def _calculate_match_confidence(
        self,
        candidate: str,
        target: str,
        variations: Dict[str, EntityVariation]
    ) -> Tuple[float, str]:
        """Enhanced confidence calculation with ASR-specific patterns"""
        
        # Check known misrecognitions first
        candidate_lower = candidate.lower()
        target_lower = target.lower()
        
        for wrong, correct in self.common_misrecognitions.items():
            if candidate_lower == wrong and target_lower == correct:
                return 0.95, "known_asr_error"
        
        # Check for phonetic similarity in endings
        # "Mecano" sounds like end of "Upamecano"
        if len(candidate) >= 5 and len(target) >= 8:
            if target_lower[-len(candidate_lower):] == candidate_lower:
                # This is a strong signal for ASR truncation
                freq_ratio = variations[target].frequency / variations[candidate].frequency
                if freq_ratio >= 2:
                    return 0.85, "phonetic_suffix_match"
        
        # Fall back to parent class logic
        return super()._calculate_match_confidence(candidate, target, variations)