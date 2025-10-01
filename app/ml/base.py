"""Base ML model abstractions"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import hashlib
import time
import logging

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Types of ML models in the pipeline"""
    PUNCTUATION = "punctuation"
    GRAMMAR = "grammar"
    FILLER = "filler"
    SCORER = "scorer"


class EditType(str, Enum):
    """Allowed edit types for ML models"""
    INSERT_PUNCTUATION = "insert_punctuation"
    REMOVE_PUNCTUATION = "remove_punctuation"
    CHANGE_CASE = "change_case"
    INSERT_ARTICLE = "insert_article"
    REMOVE_ARTICLE = "remove_article"
    FIX_AGREEMENT = "fix_agreement"
    FIX_APOSTROPHE = "fix_apostrophe"
    REMOVE_FILLER = "remove_filler"
    FIX_SPACING = "fix_spacing"
    FIX_DIACRITIC = "fix_diacritic"


@dataclass
class MLConfig:
    """Configuration for ML models"""
    model_type: ModelType
    model_path: Optional[str] = None
    language: str = "en"
    device: str = "cpu"  # cpu or cuda
    batch_size: int = 1
    max_length: int = 512
    temperature: float = 0.0  # Always 0 for determinism
    seed: int = 42  # Fixed seed for reproducibility
    quantized: bool = True  # Use quantized models by default
    cache_enabled: bool = True
    fallback_to_rules: bool = True


@dataclass
class MLResult:
    """Result from ML model processing"""
    original_text: str
    predicted_text: str
    confidence: float
    edits: List[Dict[str, Any]]  # List of specific edits made
    edit_types: List[EditType]
    processing_time_ms: float
    model_type: ModelType
    fallback_used: bool = False
    cache_hit: bool = False
    
    @property
    def edit_distance(self) -> int:
        """Calculate Levenshtein edit distance"""
        return self._levenshtein(self.original_text, self.predicted_text)
    
    @property
    def edit_ratio(self) -> float:
        """Calculate edit ratio (0-1)"""
        if not self.original_text:
            return 0.0
        return self.edit_distance / len(self.original_text)
    
    def _levenshtein(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row are one character longer than s2
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


class MLModel(ABC):
    """Abstract base class for ML models"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self._cache: Dict[str, MLResult] = {}
        self._model = None
        self._tokenizer = None
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Load model and prepare for inference"""
        pass
    
    @abstractmethod
    async def predict(self, text: str, context: Optional[Dict[str, Any]] = None) -> MLResult:
        """Run inference on text"""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        pass
    
    def _get_cache_key(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for text + context"""
        cache_str = f"{text}_{self.config.model_type}_{self.config.language}"
        if context:
            cache_str += f"_{str(sorted(context.items()))}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    async def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> MLResult:
        """Process text with caching and fallback"""
        if not self._initialized:
            await self.initialize()
        
        # Check cache
        if self.config.cache_enabled:
            cache_key = self._get_cache_key(text, context)
            if cache_key in self._cache:
                result = self._cache[cache_key]
                result.cache_hit = True
                return result
        
        # Run inference
        start_time = time.time()
        try:
            result = await self.predict(text, context)
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            # Cache result
            if self.config.cache_enabled:
                self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"ML model {self.config.model_type} failed: {e}")
            if self.config.fallback_to_rules:
                # Return original text as fallback
                return MLResult(
                    original_text=text,
                    predicted_text=text,
                    confidence=0.0,
                    edits=[],
                    edit_types=[],
                    processing_time_ms=(time.time() - start_time) * 1000,
                    model_type=self.config.model_type,
                    fallback_used=True,
                    cache_hit=False
                )
            raise
    
    def clear_cache(self) -> None:
        """Clear the prediction cache"""
        self._cache.clear()
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass