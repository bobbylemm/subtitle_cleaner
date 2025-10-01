"""Grammar error correction model implementation"""

import logging
from typing import Optional, Dict, Any, List
import re
import language_tool_python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration

from app.ml.base import MLModel, MLConfig, MLResult, EditType, ModelType

logger = logging.getLogger(__name__)


# Grammar correction models
GRAMMAR_MODELS = {
    "en": "vennify/t5-base-grammar-correction",  # English-specific
    "multilang": "google/mt5-small"  # Multilingual support
}


class GrammarCorrectionModel(MLModel):
    """
    Grammar error correction using T5 and LanguageTool
    
    Combines:
    - T5 model for neural grammar correction
    - LanguageTool for rule-based correction
    - Fallback to simple rules
    """
    
    def __init__(self, config: MLConfig):
        super().__init__(config)
        config.model_type = ModelType.GRAMMAR
        self._language_tool = None
        self._model = None
        self._tokenizer = None
    
    async def initialize(self) -> None:
        """Initialize the grammar model"""
        logger.info(f"Initializing grammar model for {self.config.language}")
        
        try:
            # Initialize LanguageTool for rule-based correction
            lang_map = {
                "en": "en-US",
                "es": "es",
                "fr": "fr",
                "de": "de",
                "it": "it",
                "pt": "pt",
                "nl": "nl"
            }
            
            if self.config.language in lang_map:
                self._language_tool = language_tool_python.LanguageTool(
                    lang_map[self.config.language]
                )
                logger.info(f"LanguageTool loaded for {self.config.language}")
            
            # Load transformer model for advanced correction
            if self.config.language == "en":
                model_name = GRAMMAR_MODELS["en"]
            else:
                model_name = GRAMMAR_MODELS["multilang"]
            
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = T5ForConditionalGeneration.from_pretrained(model_name)
            
            if self.config.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            
            self._model.eval()
            self._initialized = True
            logger.info(f"Grammar model loaded: {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load grammar models, using fallback: {e}")
            self._initialized = True
    
    async def predict(self, text: str, context: Optional[Dict[str, Any]] = None) -> MLResult:
        """
        Correct grammatical errors using multiple approaches
        """
        original = text
        predicted = text
        edits = []
        edit_types = []
        confidence = 0.5
        
        try:
            # Try LanguageTool first for rule-based correction
            if self._language_tool:
                matches = self._language_tool.check(text)
                
                # Apply corrections with constraints
                offset = 0
                for match in matches:
                    # Skip if no replacements or too many changes
                    if not match.replacements or len(match.replacements[0]) > len(match.matchedText) + 8:
                        continue
                    
                    # Apply first suggested replacement
                    replacement = match.replacements[0]
                    start = match.offset + offset
                    end = match.offset + match.errorLength + offset
                    
                    predicted = predicted[:start] + replacement + predicted[end:]
                    offset += len(replacement) - match.errorLength
                    
                    edits.append({
                        "type": match.ruleId,
                        "from": match.matchedText,
                        "to": replacement,
                        "message": match.message
                    })
                    
                    # Map to edit types
                    if "article" in match.ruleId.lower():
                        edit_types.append(EditType.INSERT_ARTICLE)
                    elif "apostrophe" in match.ruleId.lower() or "contraction" in match.ruleId.lower():
                        edit_types.append(EditType.FIX_APOSTROPHE)
                    else:
                        edit_types.append(EditType.FIX_VERB_AGREEMENT)
                
                confidence = 0.8
            
            # Try transformer model for more advanced correction
            elif self._model and self._tokenizer:
                with torch.no_grad():
                    # Prepare input
                    input_text = f"grammar: {text}"
                    inputs = self._tokenizer(
                        input_text,
                        max_length=256,
                        truncation=True,
                        padding=True,
                        return_tensors="pt"
                    )
                    
                    if self.config.device == "cuda" and torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    # Generate correction
                    outputs = self._model.generate(
                        **inputs,
                        max_length=256,
                        temperature=0.0,  # Deterministic
                        do_sample=False
                    )
                    
                    predicted = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Track changes
                    if predicted != original:
                        edits.append({
                            "type": "neural_correction",
                            "from": original,
                            "to": predicted
                        })
                        edit_types.append(EditType.FIX_VERB_AGREEMENT)
                    
                    confidence = 0.75
            
            else:
                # Fallback to rule-based approach
                predicted, rule_edits, rule_types = self._apply_rules(text)
                edits = rule_edits
                edit_types = rule_types
                confidence = 0.6
                
        except Exception as e:
            logger.warning(f"Grammar correction failed, using fallback: {e}")
            predicted, rule_edits, rule_types = self._apply_rules(text)
            edits = rule_edits
            edit_types = rule_types
            confidence = 0.5
        
        return MLResult(
            original_text=original,
            predicted_text=predicted,
            confidence=confidence,
            edits=edits,
            edit_types=edit_types,
            processing_time_ms=0.0,
            model_type=self.config.model_type,
            fallback_used=(not self._language_tool and not self._model),
            cache_hit=False
        )
    
    def _apply_rules(self, text: str) -> tuple:
        """Apply simple rule-based corrections as fallback"""
        predicted = text
        edits = []
        edit_types = []
        
        # Fix a/an articles
        predicted = re.sub(r'\ba\s+([aeiouAEIOU])', r'an \1', predicted)
        predicted = re.sub(r'\ban\s+([^aeiouAEIOU\s])', r'a \1', predicted)
        
        if 'an ' in predicted and 'an ' not in text:
            edits.append({"type": "article_correction", "from": "a", "to": "an"})
            edit_types.append(EditType.INSERT_ARTICLE)
        
        # Fix common contractions
        contractions = {
            "dont": "don't", "wont": "won't", "cant": "can't",
            "shouldnt": "shouldn't", "wouldnt": "wouldn't",
            "couldnt": "couldn't", "didnt": "didn't"
        }
        
        for wrong, correct in contractions.items():
            if f" {wrong} " in predicted.lower():
                pattern = re.compile(r'\b' + wrong + r'\b', re.IGNORECASE)
                predicted = pattern.sub(correct, predicted)
                edits.append({"type": "contraction_fix", "from": wrong, "to": correct})
                edit_types.append(EditType.FIX_APOSTROPHE)
        
        return predicted, edits, edit_types
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return ["en", "es", "fr", "de", "it", "pt", "nl"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        backend = "languagetool" if self._language_tool else ("transformer" if self._model else "rule_based")
        model_name = GRAMMAR_MODELS.get("en" if self.config.language == "en" else "multilang", "unknown")
        
        return {
            "name": "GrammarCorrectionModel",
            "type": "grammar_error_correction",
            "backend": backend,
            "model": model_name if self._model else None,
            "languages": self.get_supported_languages(),
            "quantized": self.config.quantized,
            "device": self.config.device,
            "cache_size": len(self._cache)
        }