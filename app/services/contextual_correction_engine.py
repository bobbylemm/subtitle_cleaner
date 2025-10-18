"""
Contextual Correction Engine - ML-Enhanced Subtitle Correction System
Author: World-class Staff Engineer Design
Version: 1.0.0

This module implements a three-layer cascading correction system that solves:
1. Over-correction of function words
2. Lack of semantic context in corrections
3. Word segmentation failures for compound errors
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class CorrectionMode(Enum):
    """Correction confidence modes for production tuning"""
    LEGACY = "legacy"          # Original behavior
    CONSERVATIVE = "conservative"  # High precision, lower recall
    BALANCED = "balanced"      # Balanced precision/recall
    AGGRESSIVE = "aggressive"  # High recall, may have false positives


@dataclass
class CorrectionCandidate:
    """Enhanced correction candidate with confidence factors"""
    original: str
    replacement: str
    confidence: float
    linguistic_score: float  # Based on POS and grammar
    semantic_score: float    # Based on context coherence
    statistical_score: float # Based on frequency patterns
    source: str             # Which layer generated this


class FunctionWordProtector:
    """
    Layer 1: Protects grammatical function words from incorrect replacement.
    Runs in O(1) time with minimal overhead.
    """
    
    def __init__(self):
        # Grammatical function words by category
        self.function_words = {
            'prepositions': {
                'on', 'in', 'at', 'by', 'for', 'with', 'about', 'against',
                'between', 'into', 'through', 'during', 'before', 'after',
                'above', 'below', 'to', 'from', 'up', 'down', 'out', 'off'
            },
            'articles': {'a', 'an', 'the'},
            'conjunctions': {
                'and', 'or', 'but', 'nor', 'so', 'yet', 'for', 'because',
                'although', 'since', 'unless', 'while', 'if', 'when', 'where'
            },
            'pronouns': {
                'i', 'me', 'you', 'he', 'him', 'she', 'her', 'it', 'we',
                'us', 'they', 'them', 'my', 'your', 'his', 'her', 'its',
                'our', 'their', 'this', 'that', 'these', 'those'
            },
            'auxiliaries': {
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                'could', 'may', 'might', 'must', 'can', 'shall'
            }
        }
        
        # Flatten for quick lookup
        self.all_function_words = set()
        for category_words in self.function_words.values():
            self.all_function_words.update(category_words)
        
        # Position patterns where function words are common
        self.position_patterns = {
            'sentence_start': ['the', 'a', 'an', 'this', 'that', 'these', 'those'],
            'before_noun': ['the', 'a', 'an', 'this', 'that', 'my', 'your', 'his', 'her'],
            'between_words': ['and', 'or', 'but', 'of', 'in', 'on', 'at', 'to']
        }
    
    def should_protect(self, word: str, context: List[str], position: int) -> bool:
        """
        Determine if a word should be protected from replacement.
        
        Args:
            word: The word to check
            context: Surrounding words
            position: Position in sentence
            
        Returns:
            True if word should be protected, False otherwise
        """
        word_lower = word.lower()
        
        # Check if it's a function word
        if word_lower not in self.all_function_words:
            return False
        
        # Additional context checks
        if position == 0:  # Sentence start
            # Some function words can start sentences
            if word_lower in self.position_patterns['sentence_start']:
                return True
        
        # Check surrounding context
        if position > 0 and position < len(context) - 1:
            prev_word = context[position - 1].lower() if position > 0 else ''
            next_word = context[position + 1].lower() if position < len(context) - 1 else ''
            
            # Pattern: [verb] + [preposition] + [noun]
            # Don't replace the preposition
            if word_lower in self.function_words['prepositions']:
                return True
            
            # Pattern: [article] + [adjective/noun]
            if word_lower in self.function_words['articles'] and next_word:
                return True
        
        return True  # Default to protecting function words
    
    def get_protection_score(self, word: str) -> float:
        """
        Get a protection score for a word (0.0 = no protection, 1.0 = full protection).
        """
        word_lower = word.lower()
        
        if word_lower in self.function_words['articles']:
            return 1.0  # Never replace articles
        elif word_lower in self.function_words['prepositions']:
            return 0.95  # Very rarely replace prepositions
        elif word_lower in self.function_words['conjunctions']:
            return 0.9
        elif word_lower in self.function_words['pronouns']:
            return 0.85
        elif word_lower in self.function_words['auxiliaries']:
            return 0.8
        else:
            return 0.0  # No protection


class SemanticDisambiguator:
    """
    Layer 2: Uses context windows and domain knowledge for semantic disambiguation.
    Handles cases like 'contrast' vs 'contract' based on surrounding context.
    """
    
    def __init__(self):
        # Domain-specific context patterns
        self.domain_contexts = {
            'sports': {
                'keywords': ['player', 'team', 'coach', 'match', 'game', 'transfer', 
                            'sign', 'deal', 'club', 'manager', 'football', 'soccer'],
                'expected_terms': {
                    'contract': ['sign', 'deal', 'transfer', 'player', 'club'],
                    'manager': ['team', 'club', 'football', 'coach'],
                    'transfer': ['player', 'club', 'deal', 'sign']
                }
            },
            'business': {
                'keywords': ['company', 'revenue', 'market', 'business', 'CEO', 'investment'],
                'expected_terms': {
                    'contract': ['sign', 'deal', 'agreement', 'company'],
                    'investment': ['market', 'revenue', 'business', 'company']
                }
            }
        }
        
        # Common confusables with context clues
        self.confusables = {
            ('contrast', 'contract'): {
                'contract_indicators': ['sign', 'deal', 'agreement', 'negotiate', 'extend'],
                'contrast_indicators': ['compare', 'difference', 'unlike', 'however']
            },
            ('then', 'than'): {
                'than_indicators': ['more', 'less', 'better', 'worse', 'greater'],
                'then_indicators': ['after', 'next', 'subsequently', 'time']
            }
        }
    
    def disambiguate(self, word: str, context: List[str], window_size: int = 7) -> Tuple[str, float]:
        """
        Disambiguate a word based on semantic context.
        
        Args:
            word: The word to disambiguate
            context: Full text context
            window_size: Size of context window to consider
            
        Returns:
            Tuple of (corrected_word, confidence_score)
        """
        word_lower = word.lower()
        
        # Extract context window
        context_words = self._extract_context_window(word, context, window_size)
        
        # Check for known confusables
        for confusable_pair, indicators in self.confusables.items():
            if word_lower in confusable_pair:
                other_word = confusable_pair[0] if confusable_pair[1] == word_lower else confusable_pair[1]
                
                # Score each option based on context
                word_score = self._score_context_match(word_lower, context_words, indicators)
                other_score = self._score_context_match(other_word, context_words, indicators)
                
                if other_score > word_score + 0.3:  # Significant difference threshold
                    return other_word, other_score
        
        # Check domain-specific expectations
        domain = self._detect_domain(context_words)
        if domain:
            expected_word = self._check_domain_expectations(word_lower, context_words, domain)
            if expected_word and expected_word != word_lower:
                return expected_word, 0.8
        
        return word, 0.0  # No change needed
    
    def _extract_context_window(self, word: str, context: List[str], window_size: int) -> List[str]:
        """Extract surrounding context words."""
        try:
            word_idx = context.index(word)
            start_idx = max(0, word_idx - window_size // 2)
            end_idx = min(len(context), word_idx + window_size // 2 + 1)
            return context[start_idx:end_idx]
        except ValueError:
            return context[:window_size]
    
    def _score_context_match(self, word: str, context: List[str], indicators: Dict) -> float:
        """Score how well a word matches its expected context."""
        score = 0.0
        key = f"{word}_indicators"
        
        if key in indicators:
            expected_context = indicators[key]
            context_lower = [w.lower() for w in context]
            
            for indicator in expected_context:
                if indicator in context_lower:
                    score += 0.2
            
        return min(score, 1.0)
    
    def _detect_domain(self, context: List[str]) -> Optional[str]:
        """Detect the domain of the text based on keywords."""
        context_lower = [w.lower() for w in context]
        domain_scores = {}
        
        for domain, domain_data in self.domain_contexts.items():
            score = sum(1 for keyword in domain_data['keywords'] if keyword in context_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return None
    
    def _check_domain_expectations(self, word: str, context: List[str], domain: str) -> Optional[str]:
        """Check if word matches domain expectations."""
        if domain not in self.domain_contexts:
            return None
        
        expected_terms = self.domain_contexts[domain]['expected_terms']
        context_lower = [w.lower() for w in context]
        
        for expected_word, context_clues in expected_terms.items():
            if any(clue in context_lower for clue in context_clues):
                # Check if current word might be a misrecognition
                if self._is_similar(word, expected_word):
                    return expected_word
        
        return None
    
    def _is_similar(self, word1: str, word2: str) -> bool:
        """Check if two words are phonetically or orthographically similar."""
        # Simple edit distance check
        if len(word1) == 0 or len(word2) == 0:
            return False
        
        # If words share first letter and are similar length
        if word1[0] == word2[0] and abs(len(word1) - len(word2)) <= 2:
            common_chars = sum(1 for c in word1 if c in word2)
            if common_chars / max(len(word1), len(word2)) > 0.6:
                return True
        
        return False


class CompoundWordSegmenter:
    """
    Layer 3: Handles compound word errors from accent/pronunciation issues.
    Uses dynamic programming for optimal word boundary detection.
    """
    
    def __init__(self, dictionary_path: Optional[str] = None):
        # Load dictionary (in production, use a proper dictionary file)
        self.dictionary = self._load_dictionary(dictionary_path)
        
        # Common compound errors from accent issues
        self.known_compounds = {
            'thebal': 'the verbal',
            'theball': 'the ball',
            'inthe': 'in the',
            'ofthe': 'of the',
            'tothe': 'to the',
            'andthe': 'and the',
            'forthe': 'for the',
            'withthe': 'with the'
        }
        
        # Cache for segmentation results
        self.segmentation_cache = {}
    
    def _load_dictionary(self, path: Optional[str]) -> Set[str]:
        """Load dictionary of valid words."""
        # Basic dictionary for demo - in production, use comprehensive dictionary
        basic_dict = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
            'verbal', 'agreement', 'contract', 'deal', 'ball', 'game', 'player'
        }
        
        if path:
            try:
                with open(path, 'r') as f:
                    return set(word.strip().lower() for word in f)
            except Exception:
                logger.warning(f"Could not load dictionary from {path}, using basic dictionary")
        
        return basic_dict
    
    def segment(self, word: str) -> Tuple[str, float]:
        """
        Segment a potentially compound word into components.
        
        Args:
            word: The word to segment
            
        Returns:
            Tuple of (segmented version, confidence score)
        """
        word_lower = word.lower()
        
        # Check known compounds first
        if word_lower in self.known_compounds:
            return self.known_compounds[word_lower], 0.95
        
        # Check cache
        if word_lower in self.segmentation_cache:
            return self.segmentation_cache[word_lower]
        
        # If word is in dictionary, no segmentation needed
        if word_lower in self.dictionary:
            return word, 0.0
        
        # Try dynamic programming segmentation
        segmentation = self._dp_segment(word_lower)
        
        if segmentation and len(segmentation) > 1:
            result = ' '.join(segmentation)
            confidence = self._calculate_segmentation_confidence(segmentation)
            self.segmentation_cache[word_lower] = (result, confidence)
            return result, confidence
        
        return word, 0.0
    
    def _dp_segment(self, word: str) -> List[str]:
        """
        Use dynamic programming to find optimal word segmentation.
        """
        n = len(word)
        if n == 0:
            return []
        
        # dp[i] = (cost, segmentation) for word[:i]
        dp = [(float('inf'), [])] * (n + 1)
        dp[0] = (0, [])
        
        for i in range(1, n + 1):
            # Try all possible last words
            for j in range(max(0, i - 15), i):  # Max word length 15
                substring = word[j:i]
                
                if substring in self.dictionary:
                    cost = dp[j][0] + 1  # Cost is number of words
                    if cost < dp[i][0]:
                        dp[i] = (cost, dp[j][1] + [substring])
                elif len(substring) == 1:
                    # Single character as last resort
                    cost = dp[j][0] + 10  # High penalty for single chars
                    if cost < dp[i][0]:
                        dp[i] = (cost, dp[j][1] + [substring])
        
        return dp[n][1] if dp[n][0] < float('inf') else [word]
    
    def _calculate_segmentation_confidence(self, segments: List[str]) -> float:
        """Calculate confidence score for a segmentation."""
        if not segments:
            return 0.0
        
        # Higher confidence for common word combinations
        confidence = 0.5
        
        # Boost if all segments are dictionary words
        if all(seg in self.dictionary for seg in segments):
            confidence += 0.3
        
        # Boost for common patterns
        if len(segments) == 2:
            if segments[0] in ['the', 'a', 'an'] and segments[1] in self.dictionary:
                confidence += 0.15
            elif segments[0] in ['in', 'on', 'at', 'to', 'for'] and segments[1] in ['the']:
                confidence += 0.15
        
        return min(confidence, 0.95)


class ContextualCorrectionEngine:
    """
    Main orchestrator for the three-layer correction system.
    Integrates all layers with confidence scoring and mode selection.
    """
    
    def __init__(self, mode: CorrectionMode = CorrectionMode.BALANCED):
        self.mode = mode
        self.protector = FunctionWordProtector()
        self.disambiguator = SemanticDisambiguator()
        self.segmenter = CompoundWordSegmenter()
        
        # Confidence thresholds by mode
        self.thresholds = {
            CorrectionMode.CONSERVATIVE: {
                'function_word': 0.95,
                'semantic': 0.85,
                'segmentation': 0.9,
                'overall': 0.85
            },
            CorrectionMode.BALANCED: {
                'function_word': 0.85,
                'semantic': 0.75,
                'segmentation': 0.8,
                'overall': 0.75
            },
            CorrectionMode.AGGRESSIVE: {
                'function_word': 0.7,
                'semantic': 0.6,
                'segmentation': 0.7,
                'overall': 0.6
            },
            CorrectionMode.LEGACY: {
                'function_word': 0.0,
                'semantic': 0.0,
                'segmentation': 0.0,
                'overall': 0.0
            }
        }
        
        self.stats = defaultdict(int)
    
    def process_corrections(
        self,
        text: str,
        proposed_corrections: Dict[str, str],
        context_entities: Dict[str, any]
    ) -> Dict[str, str]:
        """
        Process proposed corrections through all three layers.
        
        Args:
            text: Full text being corrected
            proposed_corrections: Initial corrections from statistical analysis
            context_entities: Entities from context extraction
            
        Returns:
            Filtered and enhanced corrections
        """
        if self.mode == CorrectionMode.LEGACY:
            return proposed_corrections
        
        words = text.split()
        filtered_corrections = {}
        thresholds = self.thresholds[self.mode]
        
        # Process each proposed correction
        for original, replacement in proposed_corrections.items():
            confidence_scores = {}
            
            # Layer 1: Function word protection
            if len(original) <= 3:  # Short words are often function words
                protection_score = self.protector.get_protection_score(original)
                if protection_score > thresholds['function_word']:
                    self.stats['protected_function_words'] += 1
                    continue  # Skip this correction
                confidence_scores['function'] = 1.0 - protection_score
            else:
                confidence_scores['function'] = 1.0
            
            # Layer 2: Semantic disambiguation
            semantic_result, semantic_confidence = self.disambiguator.disambiguate(
                original, words, window_size=7
            )
            if semantic_result != original:
                if semantic_confidence > thresholds['semantic']:
                    replacement = semantic_result
                    confidence_scores['semantic'] = semantic_confidence
                    self.stats['semantic_corrections'] += 1
            else:
                confidence_scores['semantic'] = 0.5  # Neutral
            
            # Layer 3: Compound word segmentation (for unrecognized words)
            if original not in self.segmenter.dictionary:
                segmented, seg_confidence = self.segmenter.segment(original)
                if segmented != original and seg_confidence > thresholds['segmentation']:
                    replacement = segmented
                    confidence_scores['segmentation'] = seg_confidence
                    self.stats['segmentation_corrections'] += 1
            
            # Calculate overall confidence
            if confidence_scores:
                overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
                
                if overall_confidence >= thresholds['overall']:
                    filtered_corrections[original] = replacement
                    self.stats['total_corrections'] += 1
                else:
                    self.stats['rejected_corrections'] += 1
        
        # Log statistics
        logger.info(f"Correction statistics: {dict(self.stats)}")
        
        return filtered_corrections
    
    def analyze_correction_quality(
        self,
        original_text: str,
        corrected_text: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Analyze the quality of corrections for monitoring and tuning.
        """
        metrics = {
            'edit_distance': self._calculate_edit_distance(original_text, corrected_text),
            'word_accuracy': self._calculate_word_accuracy(original_text, corrected_text),
            'function_word_preservation': self._calculate_function_preservation(original_text, corrected_text)
        }
        
        if ground_truth:
            metrics['accuracy'] = self._calculate_accuracy(corrected_text, ground_truth)
        
        return metrics
    
    def _calculate_edit_distance(self, text1: str, text2: str) -> float:
        """Calculate normalized edit distance between texts."""
        # Simplified edit distance calculation
        words1 = text1.split()
        words2 = text2.split()
        
        changes = sum(1 for w1, w2 in zip(words1, words2) if w1 != w2)
        return changes / max(len(words1), len(words2))
    
    def _calculate_word_accuracy(self, original: str, corrected: str) -> float:
        """Calculate word-level accuracy."""
        original_words = original.split()
        corrected_words = corrected.split()
        
        if len(original_words) != len(corrected_words):
            return 0.0
        
        correct = sum(1 for o, c in zip(original_words, corrected_words) if o == c)
        return correct / len(original_words)
    
    def _calculate_function_preservation(self, original: str, corrected: str) -> float:
        """Calculate how well function words were preserved."""
        original_words = original.split()
        corrected_words = corrected.split()
        
        preserved = 0
        total_function_words = 0
        
        for i, word in enumerate(original_words):
            if word.lower() in self.protector.all_function_words:
                total_function_words += 1
                if i < len(corrected_words) and corrected_words[i] == word:
                    preserved += 1
        
        return preserved / total_function_words if total_function_words > 0 else 1.0
    
    def _calculate_accuracy(self, predicted: str, ground_truth: str) -> float:
        """Calculate accuracy against ground truth."""
        pred_words = predicted.split()
        truth_words = ground_truth.split()
        
        correct = sum(1 for p, t in zip(pred_words, truth_words) if p == t)
        return correct / max(len(pred_words), len(truth_words))


# Integration function for existing codebase
def integrate_contextual_engine(
    document_normalizer,
    corrections: Dict[str, str],
    text: str,
    context_entities: Dict[str, any],
    mode: str = "balanced"
) -> Dict[str, str]:
    """
    Integration point for existing DocumentNormalizer.
    
    This function wraps the new contextual engine for easy integration.
    """
    # Convert mode string to enum
    mode_map = {
        'legacy': CorrectionMode.LEGACY,
        'conservative': CorrectionMode.CONSERVATIVE,
        'balanced': CorrectionMode.BALANCED,
        'aggressive': CorrectionMode.AGGRESSIVE
    }
    
    correction_mode = mode_map.get(mode, CorrectionMode.BALANCED)
    
    # Create engine
    engine = ContextualCorrectionEngine(mode=correction_mode)
    
    # Process corrections through the engine
    filtered_corrections = engine.process_corrections(
        text=text,
        proposed_corrections=corrections,
        context_entities=context_entities
    )
    
    return filtered_corrections