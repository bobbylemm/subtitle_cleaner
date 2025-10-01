"""Punctuation and truecasing model implementation"""

import logging
from typing import Optional, Dict, Any, List
import re
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    pipeline
)

from app.ml.base import MLModel, MLConfig, MLResult, EditType

logger = logging.getLogger(__name__)

# Use lightweight models for punctuation restoration
PUNCT_MODELS = {
    "multilang": "oliverguhr/fullstop-punctuation-multilang-large",
    "en": "Qishuai/distilbert_punctuator_en",  # Smaller for English
}


class PunctuationModel(MLModel):
    """
    Real punctuation and capitalization restoration using transformers
    
    Uses efficient models that work on CPU:
    - oliverguhr/fullstop-punctuation-multilang-large (multi-language)
    - Qishuai/distilbert_punctuator_en (English only, smaller)
    """
    
    async def initialize(self) -> None:
        """Initialize the punctuation model"""
        logger.info(f"Loading punctuation model for {self.config.language}")
        
        try:
            # Choose model based on language and device
            if self.config.language == "en" and self.config.device == "cpu":
                model_name = PUNCT_MODELS["en"]
            else:
                model_name = PUNCT_MODELS["multilang"]
            
            # Load tokenizer and model
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            # Move to device if GPU available
            if self.config.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            
            # Set to eval mode
            self._model.eval()
            
            # Create pipeline for inference
            self._pipeline = pipeline(
                "token-classification",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if self.config.device == "cuda" and torch.cuda.is_available() else -1
            )
            
            self._initialized = True
            logger.info(f"Punctuation model loaded: {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load model, using fallback: {e}")
            self._initialized = True
            self._pipeline = None
    
    async def predict(self, text: str, context: Optional[Dict[str, Any]] = None) -> MLResult:
        """
        Predict punctuation and capitalization using transformer models
        """
        original = text
        predicted = text
        edits = []
        edit_types = []
        
        try:
            if self._pipeline and self._initialized:
                # Use transformer model for prediction
                with torch.no_grad():
                    # Run the pipeline
                    results = self._pipeline(text)
                    
                    # Process token classifications
                    tokens = []
                    for token_result in results:
                        word = token_result['word']
                        label = token_result['entity']
                        
                        # Handle punctuation insertion
                        if label in ['PERIOD', 'COMMA', 'QUESTION', 'EXCLAMATION']:
                            punct_map = {
                                'PERIOD': '.',
                                'COMMA': ',',
                                'QUESTION': '?',
                                'EXCLAMATION': '!'
                            }
                            punct = punct_map.get(label, '')
                            if punct and not word.endswith(punct):
                                word += punct
                                edits.append({
                                    "type": "insert_punctuation",
                                    "text": punct
                                })
                                edit_types.append(EditType.INSERT_PUNCTUATION)
                        
                        # Handle capitalization
                        if label == 'CAPITALIZE' and word[0].islower():
                            word = word[0].upper() + word[1:]
                            edits.append({
                                "type": "fix_capitalization",
                                "text": word
                            })
                            edit_types.append(EditType.FIX_CAPITALIZATION)
                        
                        tokens.append(word)
                    
                    # Reconstruct text from tokens
                    predicted = self._detokenize(tokens)
                    
                    # Calculate confidence from model scores
                    avg_score = sum(r.get('score', 0.5) for r in results) / len(results) if results else 0.5
                    confidence = avg_score
                    
            else:
                # Fallback to rule-based approach
                predicted, edits, edit_types = self._apply_rules(text)
                confidence = 0.7
                
        except Exception as e:
            logger.warning(f"Model prediction failed, using fallback: {e}")
            predicted, edits, edit_types = self._apply_rules(text)
            confidence = 0.6
        
        return MLResult(
            original_text=original,
            predicted_text=predicted,
            confidence=confidence,
            edits=edits,
            edit_types=edit_types,
            processing_time_ms=0.0,  # Will be set by base class
            model_type=self.config.model_type,
            fallback_used=(not self._pipeline),
            cache_hit=False
        )
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        # ByT5 supports 100+ languages
        return ["en", "es", "fr", "de", "it", "pt", "nl"]
    
    def _apply_rules(self, text: str) -> tuple:
        """Apply rule-based punctuation restoration as fallback"""
        predicted = text
        edits = []
        edit_types = []
        
        # 1. Capitalize first letter of sentences
        predicted = re.sub(r'^([a-z])', lambda m: m.group(1).upper(), predicted)
        predicted = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), predicted)
        
        # 2. Add missing periods at end
        if predicted and predicted[-1].isalnum():
            predicted += '.'
            edits.append({
                "type": "insert_punctuation",
                "position": len(predicted) - 1,
                "text": "."
            })
            edit_types.append(EditType.INSERT_PUNCTUATION)
        
        # 3. Fix spacing around punctuation
        predicted = re.sub(r'\s+([.,!?;:])', r'\1', predicted)  # Remove space before
        predicted = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', predicted)  # Add space after
        
        # 4. Capitalize 'I'
        predicted = re.sub(r'\bi\b', 'I', predicted)
        
        return predicted, edits, edit_types
    
    def _detokenize(self, tokens: List[str]) -> str:
        """Convert tokens back to text with proper spacing"""
        if not tokens:
            return ""
        
        result = []
        for i, token in enumerate(tokens):
            # Handle subword tokens
            if token.startswith('##'):
                result.append(token[2:])
            elif i == 0:
                result.append(token)
            elif token in '.,!?;:':
                result.append(token)
            else:
                result.append(' ' + token)
        
        return ''.join(result)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        backend = "transformer" if self._pipeline else "rule_based"
        model_name = PUNCT_MODELS.get("en" if self.config.language == "en" else "multilang", "unknown")
        
        return {
            "name": "PunctuationModel",
            "type": "punctuation_restoration",
            "backend": backend,
            "model": model_name,
            "languages": self.get_supported_languages(),
            "quantized": self.config.quantized,
            "device": self.config.device,
            "cache_size": len(self._cache)
        }