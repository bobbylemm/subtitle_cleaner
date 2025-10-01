"""ONNX-based punctuation restoration and truecasing model"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import hashlib
import json

logger = logging.getLogger(__name__)


class ONNXPunctuationModel:
    """
    Lightweight punctuation restoration and truecasing using ONNX
    
    Uses pre-trained models for deterministic, fast inference
    Handles punctuation, capitalization, and sentence boundaries in one pass
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        "en": {
            "model_id": "1-800-BAD-CODE/punctuation_fullstop_truecase_english",
            "model_url": "https://huggingface.co/1-800-BAD-CODE/punctuation_fullstop_truecase_english/resolve/main/model.onnx",
            "max_length": 256,
            "window_size": 200,  # Process in chunks
            "overlap": 50  # Overlap between chunks
        }
    }
    
    # Punctuation tokens
    PUNCT_MAPPING = {
        "PERIOD": ".",
        "COMMA": ",", 
        "QUESTION": "?",
        "EXCLAMATION": "!",
        "COLON": ":",
        "SEMICOLON": ";",
        "DASH": "-",
        "NONE": ""
    }
    
    def __init__(self, language: str = "en", cache_dir: str = "models/punctuation"):
        self.language = language
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = self.MODEL_CONFIGS.get(language, self.MODEL_CONFIGS["en"])
        self.model_path = self.cache_dir / f"punct_{language}.onnx"
        
        self.session = None
        self.tokenizer = None
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize model and tokenizer"""
        if self._initialized:
            return
            
        try:
            # Load or download model
            if not self.model_path.exists():
                logger.info(f"Model not found locally, using fallback")
                # In production, download from model_url
                # For now, we'll use a simple rule-based fallback
                self._use_fallback = True
            else:
                # Load ONNX model
                providers = ['CPUExecutionProvider']
                self.session = ort.InferenceSession(
                    str(self.model_path),
                    providers=providers
                )
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config["model_id"],
                    local_files_only=True
                )
                self._use_fallback = False
                
            self._initialized = True
            logger.info(f"Punctuation model initialized for {self.language}")
            
        except Exception as e:
            logger.warning(f"Failed to load ONNX model, using rule-based fallback: {e}")
            self._use_fallback = True
            self._initialized = True
            
    def restore(self, text: str, segment_context: Optional[List[str]] = None) -> str:
        """
        Restore punctuation and truecasing to text
        
        Args:
            text: Input text (possibly lowercase, no punctuation)
            segment_context: Optional context from surrounding segments
            
        Returns:
            Text with restored punctuation and proper casing
        """
        if not self._initialized:
            self.initialize()
            
        if self._use_fallback:
            return self._rule_based_restoration(text)
            
        try:
            # Use ONNX model for restoration
            return self._model_based_restoration(text, segment_context)
        except Exception as e:
            logger.warning(f"Model inference failed, using fallback: {e}")
            return self._rule_based_restoration(text)
            
    def _model_based_restoration(self, text: str, context: Optional[List[str]] = None) -> str:
        """Use ONNX model for restoration"""
        
        # Handle empty input
        if not text or not text.strip():
            return text
            
        # Prepare input
        processed_text = text.lower().strip()
        
        # Add context if available
        if context:
            # Add previous segment for better context
            if len(context) > 0:
                processed_text = context[-1] + " " + processed_text
                
        # Process in chunks for long text
        if len(processed_text.split()) > self.config["window_size"]:
            return self._process_long_text(processed_text)
            
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            return_tensors="np",
            max_length=self.config["max_length"],
            truncation=True,
            padding=True
        )
        
        # Run inference
        outputs = self.session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        })
        
        # Decode predictions
        result = self._decode_predictions(outputs, inputs["input_ids"])
        
        # Remove context if it was added
        if context and len(context) > 0:
            # Remove the context part
            context_words = len(context[-1].split())
            result_words = result.split()
            result = " ".join(result_words[context_words:])
            
        return result
        
    def _process_long_text(self, text: str) -> str:
        """Process long text in overlapping chunks"""
        words = text.split()
        window = self.config["window_size"]
        overlap = self.config["overlap"]
        step = window - overlap
        
        results = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + window])
            restored = self._model_based_restoration(chunk, None)
            
            if i == 0:
                results.append(restored)
            else:
                # Merge with overlap handling
                restored_words = restored.split()
                results.append(" ".join(restored_words[overlap:]))
                
        return " ".join(results)
        
    def _decode_predictions(self, outputs: List[np.ndarray], input_ids: np.ndarray) -> str:
        """Decode model outputs to text with punctuation and casing"""
        
        # This is a simplified decoder
        # In practice, you'd need the specific decoding logic for your model
        punct_predictions = outputs[0]  # Punctuation logits
        case_predictions = outputs[1] if len(outputs) > 1 else None  # Casing logits
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        result = []
        
        for i, token in enumerate(tokens):
            if token in ["[PAD]", "[CLS]", "[SEP]"]:
                continue
                
            # Apply casing if predictions available
            if case_predictions is not None:
                if case_predictions[0][i] > 0.5:  # Threshold for capitalization
                    token = token.capitalize()
                    
            result.append(token)
            
            # Add punctuation after token if predicted
            if i < len(punct_predictions[0]):
                punct_idx = np.argmax(punct_predictions[0][i])
                punct = list(self.PUNCT_MAPPING.values())[punct_idx]
                if punct and punct != "NONE":
                    result.append(punct)
                    
        # Convert tokens back to text
        text = self.tokenizer.convert_tokens_to_string(result)
        return text
        
    def _rule_based_restoration(self, text: str) -> str:
        """
        Fallback rule-based punctuation restoration
        Fast, deterministic, and works offline
        """
        
        if not text or not text.strip():
            return text
            
        # Basic sentence boundary detection
        sentences = []
        current = []
        words = text.split()
        
        for i, word in enumerate(words):
            current.append(word)
            
            # Detect sentence boundaries
            if self._is_sentence_end(word, i, words):
                sentence = " ".join(current)
                # Capitalize first word
                sentence = self._capitalize_sentence(sentence)
                # Add period if missing
                if not re.search(r'[.!?]$', sentence):
                    sentence += "."
                sentences.append(sentence)
                current = []
                
        # Handle remaining words
        if current:
            sentence = " ".join(current)
            sentence = self._capitalize_sentence(sentence)
            if not re.search(r'[.!?]$', sentence):
                sentence += "."
            sentences.append(sentence)
            
        result = " ".join(sentences)
        
        # Post-processing
        result = self._fix_punctuation_spacing(result)
        result = self._capitalize_proper_nouns(result)
        
        return result
        
    def _is_sentence_end(self, word: str, index: int, words: List[str]) -> bool:
        """Heuristic to detect sentence boundaries"""
        
        # Already has sentence-ending punctuation
        if re.search(r'[.!?]$', word):
            return True
            
        # Common sentence-ending patterns
        if index < len(words) - 1:
            next_word = words[index + 1]
            
            # Next word starts with capital (likely new sentence)
            if next_word and next_word[0].isupper() and index > 0:
                # But not if current word is very short (might be abbreviation)
                if len(word) > 2:
                    return True
                    
            # Conjunctions that often start new sentences
            if next_word.lower() in ["but", "however", "therefore", "thus", "so"]:
                return True
                
        # Long enough to be a sentence
        if index > 0 and (index + 1) % 15 == 0:  # Roughly 15 words per sentence
            return True
            
        return False
        
    def _capitalize_sentence(self, sentence: str) -> str:
        """Capitalize the first letter of a sentence"""
        sentence = sentence.strip()
        if sentence:
            # Handle quotes and other starting punctuation
            for i, char in enumerate(sentence):
                if char.isalpha():
                    return sentence[:i] + char.upper() + sentence[i+1:]
        return sentence
        
    def _fix_punctuation_spacing(self, text: str) -> str:
        """Fix spacing around punctuation"""
        
        # Remove space before punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        # Add space after punctuation (except at end)
        text = re.sub(r'([,.!?;:])(?=[A-Za-z])', r'\1 ', text)
        
        # Fix multiple punctuation
        text = re.sub(r'\.{2,}', '...', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'!{2,}', '!', text)
        
        # Fix quotes
        text = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', text)
        
        return text
        
    def _capitalize_proper_nouns(self, text: str) -> str:
        """Basic proper noun capitalization"""
        
        # Common proper nouns that should be capitalized
        proper_nouns = {
            "i": "I",
            "i'm": "I'm",
            "i'll": "I'll",
            "i'd": "I'd",
            "i've": "I've",
            "monday": "Monday",
            "tuesday": "Tuesday",
            "wednesday": "Wednesday",
            "thursday": "Thursday",
            "friday": "Friday",
            "saturday": "Saturday",
            "sunday": "Sunday",
            "january": "January",
            "february": "February",
            "march": "March",
            "april": "April",
            "may": "May",
            "june": "June",
            "july": "July",
            "august": "August",
            "september": "September",
            "october": "October",
            "november": "November",
            "december": "December"
        }
        
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:"')
            if word_lower in proper_nouns:
                # Preserve original punctuation
                punct = word[len(word.rstrip('.,!?;:"')):]
                words[i] = proper_nouns[word_lower] + punct
                
        return " ".join(words)
        
    def get_confidence(self, original: str, restored: str) -> float:
        """Calculate confidence score for the restoration"""
        
        if self._use_fallback:
            # Lower confidence for rule-based
            return 0.7
            
        # Calculate based on edit distance
        edit_distance = self._calculate_edit_distance(original.lower(), restored.lower())
        max_len = max(len(original), len(restored))
        
        if max_len == 0:
            return 1.0
            
        # More edits = lower confidence (but still relatively high)
        confidence = max(0.5, 1.0 - (edit_distance / max_len) * 0.5)
        return confidence
        
    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance"""
        if len(s1) < len(s2):
            return self._calculate_edit_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]