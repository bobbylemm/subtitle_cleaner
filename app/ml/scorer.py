"""Perplexity scorer for quality assessment"""

import logging
import math
from typing import Optional, Dict, Any, List
import re
import asyncio
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from app.ml.base import MLModel, MLConfig, MLResult, ModelType

logger = logging.getLogger(__name__)

# Small models for perplexity scoring
SCORER_MODELS = {
    "en": "distilbert-base-uncased",
    "multilang": "bert-base-multilingual-cased"
}


class PerplexityScorer(MLModel):
    """
    Perplexity-based text quality scorer
    
    In production, this would use:
    - KenLM 5-gram models per language
    - Small masked language models (DistilBERT, TinyBERT)
    - Cached scoring for efficiency
    """
    
    def __init__(self, config: MLConfig):
        super().__init__(config)
        config.model_type = ModelType.SCORER
        self.ngram_scores = {}  # Cache for n-gram probabilities
        self._model = None
        self._tokenizer = None
    
    async def initialize(self) -> None:
        """Initialize the scoring model"""
        logger.info(f"Initializing perplexity scorer for {self.config.language}")
        
        try:
            # Choose model based on language
            model_name = SCORER_MODELS["en" if self.config.language == "en" else "multilang"]
            
            # Load tokenizer and model
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(model_name)
            
            # Move to device if GPU available
            if self.config.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            
            self._model.eval()
            logger.info(f"Loaded perplexity model: {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load neural model, using n-gram fallback: {e}")
            # Initialize with common n-gram patterns as fallback
            self._init_ngram_patterns()
        
        self._initialized = True
    
    async def predict(self, text: str, context: Optional[Dict[str, Any]] = None) -> MLResult:
        """Not used for scoring - use score() instead"""
        raise NotImplementedError("Use score() method for perplexity scoring")
    
    async def score(self, text: str) -> float:
        """
        Calculate perplexity score for text
        Lower is better (more fluent)
        """
        if not text:
            return float('inf')
        
        # Try neural model first
        if self._model and self._tokenizer:
            try:
                return await self._neural_perplexity(text)
            except Exception as e:
                logger.debug(f"Neural scoring failed, using n-gram: {e}")
        
        # Fallback to n-gram scoring
        return await self._ngram_perplexity(text)
    
    async def _neural_perplexity(self, text: str) -> float:
        """Calculate perplexity using masked language model"""
        with torch.no_grad():
            # Tokenize
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            if self.config.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get model outputs
            outputs = self._model(**inputs)
            logits = outputs.logits
            
            # Calculate perplexity
            # Shift labels to align with predictions
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Perplexity is exp(loss)
            perplexity = torch.exp(loss).item()
            
            return min(perplexity, 1000.0)  # Cap at 1000
    
    async def _ngram_perplexity(self, text: str) -> float:
        """Calculate perplexity using n-gram patterns (fallback)"""
        words = text.lower().split()
        if len(words) < 2:
            return 1.0
        
        total_log_prob = 0.0
        count = 0
        
        # Bigram scoring
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            prob = self._get_ngram_probability(bigram, n=2)
            if prob > 0:
                total_log_prob += math.log(prob)
                count += 1
        
        # Trigram scoring
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            prob = self._get_ngram_probability(trigram, n=3)
            if prob > 0:
                total_log_prob += math.log(prob)
                count += 1
        
        if count == 0:
            return 100.0
        
        avg_log_prob = total_log_prob / count
        perplexity = math.exp(-avg_log_prob)
        
        return min(perplexity, 1000.0)
    
    def _init_ngram_patterns(self):
        """Initialize common n-gram patterns with probabilities"""
        # Common English bigrams (simplified)
        self.ngram_scores = {
            # High probability (common)
            "the ": 0.8,
            "of the": 0.75,
            "in the": 0.7,
            "to the": 0.65,
            "and the": 0.6,
            "it is": 0.6,
            "this is": 0.55,
            "that is": 0.5,
            "i am": 0.6,
            "you are": 0.55,
            "we are": 0.5,
            "they are": 0.5,
            "he is": 0.5,
            "she is": 0.5,
            "do you": 0.5,
            "can you": 0.45,
            "will be": 0.45,
            "would be": 0.4,
            "have been": 0.4,
            "has been": 0.4,
            
            # Medium probability
            "going to": 0.4,
            "want to": 0.35,
            "need to": 0.35,
            "able to": 0.3,
            "used to": 0.3,
            "seems to": 0.25,
            "supposed to": 0.25,
            
            # Lower probability (less common but valid)
            "sort of": 0.15,
            "kind of": 0.15,
            "a lot": 0.2,
            "a bit": 0.15,
            "a little": 0.2,
            
            # Common trigrams
            "i don't know": 0.4,
            "i don't think": 0.35,
            "what do you": 0.3,
            "how do you": 0.3,
            "going to be": 0.35,
            "it's going to": 0.3,
            "that's what i": 0.25,
            "i think it's": 0.25,
        }
    
    def _get_ngram_probability(self, ngram: str, n: int) -> float:
        """
        Get probability for an n-gram
        
        In production: Query KenLM or neural model
        For demo: Use lookup table with backoff
        """
        # Direct lookup
        if ngram in self.ngram_scores:
            return self.ngram_scores[ngram]
        
        # Partial matching for unknown ngrams
        words = ngram.split()
        
        # Check if it starts with common words
        if words[0] in ["the", "a", "an", "i", "you", "we", "they", "it"]:
            return 0.1  # Low but non-zero probability
        
        # Check if it contains function words
        function_words = {"is", "are", "was", "were", "be", "been", "have", "has", "had",
                         "do", "does", "did", "will", "would", "could", "should", "can"}
        if any(w in function_words for w in words):
            return 0.05
        
        # Unknown ngram
        return 0.01
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare perplexity of two text versions
        
        Returns:
            Dictionary with scores and recommendation
        """
        score1 = asyncio.run(self.score(text1))
        score2 = asyncio.run(self.score(text2))
        
        improvement = (score1 - score2) / score1 if score1 > 0 else 0
        
        return {
            "original_perplexity": score1,
            "modified_perplexity": score2,
            "improvement_ratio": improvement,
            "recommendation": "use_modified" if score2 < score1 else "use_original"
        }
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        # KenLM models available for major languages
        return ["en", "es", "fr", "de", "it", "pt", "nl"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "PerplexityScorer",
            "type": "quality_scorer",
            "backend": "masked_lm" if self._model else "ngram",
            "languages": self.get_supported_languages(),
            "ngram_patterns": len(self.ngram_scores),
            "cache_size": len(self._cache)
        }