"""Context-aware filler detection model"""

import logging
from typing import Optional, Dict, Any, List, Set
import re
import asyncio

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available, using rule-based filler detection")

from app.ml.base import MLModel, MLConfig, MLResult, EditType, ModelType
from app.domain.constants import FILLERS, Language

logger = logging.getLogger(__name__)


class ContextualFillerDetector(MLModel):
    """
    Context-aware filler word detection
    
    In production, this would use:
    - BiLSTM-CRF sequence tagger
    - DistilBERT for context understanding
    - POS tagging for disambiguation
    """
    
    def __init__(self, config: MLConfig):
        super().__init__(config)
        config.model_type = ModelType.FILLER
        self.language_fillers = self._get_language_fillers()
        self.ambiguous_words = self._get_ambiguous_words()
        self._nlp = None
    
    async def initialize(self) -> None:
        """Initialize the filler detection model"""
        logger.info(f"Initializing filler detector for {self.config.language}")
        
        try:
            if SPACY_AVAILABLE and self.config.language == "en":
                # Load spaCy model for POS tagging
                # Note: In production, download with: python -m spacy download en_core_web_sm
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                    logger.info("spaCy model loaded for context-aware filler detection")
                except:
                    # Model not downloaded, create blank pipeline
                    self._nlp = spacy.blank("en")
                    logger.info("Using blank spaCy pipeline")
            
            self._initialized = True
            
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}")
            self._initialized = True
    
    async def predict(self, text: str, context: Optional[Dict[str, Any]] = None) -> MLResult:
        """
        Detect and remove fillers with context awareness
        
        Smart detection for ambiguous words like:
        - "like" (filler vs verb/preposition)
        - "well" (filler vs adverb/noun)
        - "right" (filler vs direction/correct)
        """
        original = text
        predicted = text
        edits = []
        edit_types = []
        removed_fillers = []
        
        # Split into words while preserving punctuation
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        result_words = []
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Check if it's a potential filler
            if word_lower in self.language_fillers:
                # Check if it's ambiguous and needs context checking
                if word_lower in self.ambiguous_words:
                    if self._is_filler_in_context(word_lower, words, i):
                        # It's a filler, remove it
                        removed_fillers.append(word_lower)
                        edits.append({
                            "type": "remove_filler",
                            "word": word,
                            "position": i,
                            "context_aware": True
                        })
                        edit_types.append(EditType.REMOVE_FILLER)
                        continue
                    else:
                        # It's content, keep it
                        result_words.append(word)
                else:
                    # Unambiguous filler, check position rules
                    if self._can_remove_filler(word_lower, words, i):
                        removed_fillers.append(word_lower)
                        edits.append({
                            "type": "remove_filler",
                            "word": word,
                            "position": i,
                            "context_aware": False
                        })
                        edit_types.append(EditType.REMOVE_FILLER)
                        continue
                    else:
                        result_words.append(word)
            else:
                result_words.append(word)
        
        # Reconstruct text
        predicted = self._reconstruct_text(result_words)
        
        # Clean up any double spaces
        predicted = re.sub(r'\s+', ' ', predicted).strip()
        
        # Calculate confidence based on context checking
        num_context_checks = sum(1 for e in edits if e.get("context_aware"))
        confidence = 0.8 if num_context_checks > 0 else 0.9
        
        return MLResult(
            original_text=original,
            predicted_text=predicted,
            confidence=confidence,
            edits=edits,
            edit_types=edit_types,
            processing_time_ms=0.0,
            model_type=self.config.model_type,
            fallback_used=False,
            cache_hit=False
        )
    
    def _get_language_fillers(self) -> Set[str]:
        """Get fillers for current language"""
        try:
            lang = Language(self.config.language)
            return FILLERS.get(lang, set())
        except:
            return FILLERS.get(Language.EN, set())
    
    def _get_ambiguous_words(self) -> Set[str]:
        """Get words that need context checking"""
        return {
            "like",    # Can be verb/preposition
            "well",    # Can be adverb/noun
            "right",   # Can be direction/correct
            "mean",    # Can be verb/adjective
            "just",    # Can be adverb (recently/only)
            "so",      # Can be conjunction/intensifier
            "actually", # Sometimes needed for emphasis
            "really",  # Sometimes needed for emphasis
        }
    
    def _is_filler_in_context(self, word: str, words: List[str], position: int) -> bool:
        """
        Determine if an ambiguous word is a filler based on context
        """
        # Try spaCy POS tagging if available
        if self._nlp and SPACY_AVAILABLE:
            try:
                # Reconstruct text around the word
                start = max(0, position - 3)
                end = min(len(words), position + 4)
                context_words = words[start:end]
                context_text = ' '.join(context_words)
                
                # Process with spaCy
                doc = self._nlp(context_text)
                
                # Find the target word in the processed doc
                target_idx = position - start
                if target_idx < len(doc):
                    token = doc[target_idx]
                    
                    # Use POS tag to determine if it's a filler
                    if word == "like":
                        # VERB or ADP (preposition) = content
                        # INTJ or PART = likely filler
                        if token.pos_ in ["VERB", "ADP"]:
                            return False
                        elif token.pos_ in ["INTJ", "PART", "ADV"]:
                            return True
                    
                    elif word == "well":
                        # ADV at sentence start = likely filler
                        # ADJ or ADV mid-sentence = content
                        if token.pos_ == "INTJ":
                            return True
                        elif position == 0 or words[position-1] in [",", "."]:
                            return True
                        return False
                    
                    elif word == "so":
                        # SCONJ (subordinating conjunction) = content
                        # ADV at start = likely filler
                        if token.pos_ == "SCONJ":
                            return False
                        elif position == 0:
                            return True
                    
            except Exception as e:
                logger.debug(f"spaCy processing failed, using heuristics: {e}")
        
        # Fallback to heuristics
        prev_word = words[position - 1].lower() if position > 0 else ""
        next_word = words[position + 1].lower() if position < len(words) - 1 else ""
        prev2_word = words[position - 2].lower() if position > 1 else ""
        next2_word = words[position + 2].lower() if position < len(words) - 2 else ""
        
        # Specific rules for each ambiguous word
        if word == "like":
            # Filler if: "it's like", "you know like", ", like,"
            # Content if: "I like", "would like", "looks like"
            if prev_word in ["i", "you", "we", "they", "would", "could", "should"]:
                return False  # It's a verb
            if next_word in ["to", "that", "this", "a", "the"]:
                return True  # Likely filler
            if prev_word in ["its", "it's", ",", "know"]:
                return True  # Likely filler
            return False
        
        elif word == "well":
            # Filler if: start of sentence, after comma
            # Content if: "very well", "as well", "well done"
            if prev_word in ["", ",", "."]:
                return True  # Likely filler
            if prev_word in ["very", "as", "pretty", "quite"]:
                return False  # It's an adverb/adjective
            if next_word == "done":
                return False
            return position == 0  # Filler at start
        
        elif word == "right":
            # Filler if: ", right?", "right,"
            # Content if: "turn right", "right answer"
            if next_word in ["?", ","]:
                return True  # Tag question filler
            if prev_word in ["turn", "go", "the", "a"]:
                return False  # Direction or adjective
            return False
        
        elif word == "so":
            # Filler if: "so," at start, "and so"
            # Content if: "so that", "so much"
            if position == 0 and next_word == ",":
                return True
            if next_word in ["that", "much", "many", "far"]:
                return False
            if prev_word == "and" and next_word == ",":
                return True
            return False
        
        elif word in ["actually", "really"]:
            # Keep if providing emphasis or correction
            # Remove if just padding
            if position == 0:
                return True  # Often filler at start
            if prev_word == "," and next_word == ",":
                return True  # Parenthetical filler
            return False
        
        return False
    
    def _can_remove_filler(self, word: str, words: List[str], position: int) -> bool:
        """Check if we can safely remove an unambiguous filler"""
        # Don't remove if it's the only word
        if len(words) == 1:
            return False
        
        # Don't remove if inside quotes (simplified check)
        quote_before = any('"' in w for w in words[:position])
        quote_after = any('"' in w for w in words[position+1:])
        if quote_before and quote_after:
            return False
        
        # Safe to remove
        return True
    
    def _reconstruct_text(self, words: List[str]) -> str:
        """Reconstruct text from words, handling punctuation"""
        if not words:
            return ""
        
        result = []
        for i, word in enumerate(words):
            if i == 0:
                result.append(word)
            elif word in '.,!?;:':
                # No space before punctuation
                result.append(word)
            elif i > 0 and words[i-1] in '("\'':
                # No space after opening brackets/quotes
                result.append(word)
            elif word in ')"\'':
                # No space before closing brackets/quotes
                result.append(word)
            else:
                result.append(' ' + word)
        
        return ''.join(result)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        # Full support for English, basic for others
        return ["en", "es", "fr", "de", "it", "pt", "nl"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "ContextualFillerDetector",
            "type": "filler_detection",
            "backend": "spacy" if self._nlp else "rule_based",
            "languages": self.get_supported_languages(),
            "context_aware": True,
            "ambiguous_words": list(self.ambiguous_words),
            "cache_size": len(self._cache)
        }