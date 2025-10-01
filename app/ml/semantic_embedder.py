"""ONNX-optimized semantic embedder for fast, deterministic embeddings"""

import os
import logging
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from functools import lru_cache
import numpy as np

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SemanticEmbedder:
    """
    Fast semantic embedder with ONNX optimization and caching
    
    Falls back to sentence-transformers or TF-IDF if ONNX unavailable
    """
    
    MODELS = {
        "minilm": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "dim": 384,
            "max_length": 256,
            "onnx_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
        },
        "mpnet": {
            "name": "sentence-transformers/all-mpnet-base-v2", 
            "dim": 768,
            "max_length": 384,
            "onnx_url": None  # Would need conversion
        }
    }
    
    def __init__(self, 
                 model_type: str = "minilm",
                 cache_dir: str = "models/embeddings",
                 use_cache: bool = True,
                 normalize: bool = True):
        """
        Initialize semantic embedder
        
        Args:
            model_type: Model to use (minilm or mpnet)
            cache_dir: Directory for model and embedding cache
            use_cache: Whether to cache embeddings
            normalize: Whether to normalize embeddings (for cosine similarity)
        """
        self.model_type = model_type
        self.model_config = self.MODELS[model_type]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.normalize = normalize
        
        self.onnx_session = None
        self.tokenizer = None
        self.fallback_model = None
        self._mode = None
        
        # Embedding cache (text hash -> embedding)
        self._embedding_cache = {}
        self._cache_file = self.cache_dir / f"{model_type}_cache.pkl"
        self._load_cache()
        
        # Initialize model
        self._initialize()
        
    def _initialize(self):
        """Initialize the embedding model (ONNX, sentence-transformers, or TF-IDF)"""
        
        # Try ONNX first (fastest)
        if ONNX_AVAILABLE:
            if self._initialize_onnx():
                self._mode = "onnx"
                logger.info(f"Initialized ONNX embedder with {self.model_type}")
                return
                
        # Try sentence-transformers (good quality, slower)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            if self._initialize_sentence_transformers():
                self._mode = "sentence_transformers"
                logger.info(f"Initialized sentence-transformers embedder with {self.model_type}")
                return
                
        # Fallback to TF-IDF (fastest, lower quality)
        self._initialize_tfidf()
        self._mode = "tfidf"
        logger.warning("Using TF-IDF fallback for embeddings (lower quality)")
        
    def _initialize_onnx(self) -> bool:
        """Initialize ONNX model"""
        try:
            model_path = self.cache_dir / f"{self.model_type}.onnx"
            
            # Download or use cached model
            if not model_path.exists():
                if self.model_config["onnx_url"]:
                    logger.info(f"Downloading ONNX model to {model_path}")
                    # In production, download from URL
                    # For now, we'll return False to use fallback
                    return False
                else:
                    # Need to convert model to ONNX
                    return False
                    
            # Load ONNX model
            providers = ['CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.onnx_session = ort.InferenceSession(
                str(model_path),
                sess_options,
                providers=providers
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["name"],
                cache_dir=self.cache_dir
            )
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to initialize ONNX: {e}")
            return False
            
    def _initialize_sentence_transformers(self) -> bool:
        """Initialize sentence-transformers model"""
        try:
            self.fallback_model = SentenceTransformer(
                self.model_config["name"],
                cache_folder=str(self.cache_dir)
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize sentence-transformers: {e}")
            return False
            
    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer as ultimate fallback"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.fallback_model = TfidfVectorizer(
            max_features=self.model_config["dim"],
            sublinear_tf=True,
            use_idf=True
        )
        # Will be fitted on first batch of texts
        self._tfidf_fitted = False
        
    @lru_cache(maxsize=10000)
    def _hash_text(self, text: str) -> str:
        """Generate hash for text (for caching)"""
        return hashlib.md5(text.encode()).hexdigest()
        
    def embed(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for texts
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            
        Returns:
            Embeddings array (num_texts x embedding_dim)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
            
        # Check cache
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if self.use_cache:
            for i, text in enumerate(texts):
                text_hash = self._hash_text(text)
                if text_hash in self._embedding_cache:
                    embeddings.append(self._embedding_cache[text_hash])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    embeddings.append(None)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            embeddings = [None] * len(texts)
            
        # Generate embeddings for uncached texts
        if uncached_texts:
            if self._mode == "onnx":
                new_embeddings = self._embed_onnx(uncached_texts, batch_size)
            elif self._mode == "sentence_transformers":
                new_embeddings = self._embed_sentence_transformers(uncached_texts, batch_size)
            else:
                new_embeddings = self._embed_tfidf(uncached_texts)
                
            # Fill in embeddings and update cache
            for i, idx in enumerate(uncached_indices):
                embeddings[idx] = new_embeddings[i]
                if self.use_cache:
                    text_hash = self._hash_text(uncached_texts[i])
                    self._embedding_cache[text_hash] = new_embeddings[i]
                    
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        # Normalize if requested (for cosine similarity)
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
            
        # Save cache periodically
        if self.use_cache and len(self._embedding_cache) % 100 == 0:
            self._save_cache()
            
        return embeddings[0] if single_text else embeddings
        
    def _embed_onnx(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using ONNX"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.model_config["max_length"],
                return_tensors="np"
            )
            
            # Run inference
            outputs = self.onnx_session.run(
                None,
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                }
            )
            
            # Mean pooling
            embeddings = outputs[0]  # token embeddings
            attention_mask = inputs["attention_mask"]
            
            # Expand attention mask for broadcasting
            mask_expanded = np.expand_dims(attention_mask, -1)
            
            # Apply mask and compute mean
            masked_embeddings = embeddings * mask_expanded
            summed = np.sum(masked_embeddings, axis=1)
            counts = np.sum(mask_expanded, axis=1) + 1e-8
            mean_pooled = summed / counts
            
            all_embeddings.append(mean_pooled)
            
        return np.vstack(all_embeddings)
        
    def _embed_sentence_transformers(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Generate embeddings using sentence-transformers"""
        embeddings = self.fallback_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=False  # We'll normalize later if needed
        )
        return embeddings
        
    def _embed_tfidf(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using TF-IDF"""
        if not self._tfidf_fitted:
            # Fit on first batch
            self.fallback_model.fit(texts)
            self._tfidf_fitted = True
            
        # Transform texts
        embeddings = self.fallback_model.transform(texts).toarray()
        
        # Pad or truncate to match expected dimensions
        target_dim = self.model_config["dim"]
        if embeddings.shape[1] < target_dim:
            # Pad with zeros
            padding = np.zeros((embeddings.shape[0], target_dim - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])
        elif embeddings.shape[1] > target_dim:
            # Truncate
            embeddings = embeddings[:, :target_dim]
            
        return embeddings
        
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute similarity between embeddings
        
        If embeddings are normalized, this is cosine similarity via dot product
        """
        if len(embeddings1.shape) == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if len(embeddings2.shape) == 1:
            embeddings2 = embeddings2.reshape(1, -1)
            
        if self.normalize:
            # Already normalized, use dot product
            return np.dot(embeddings1, embeddings2.T)
        else:
            # Compute cosine similarity
            norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
            norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
            normalized1 = embeddings1 / (norms1 + 1e-8)
            normalized2 = embeddings2 / (norms2 + 1e-8)
            return np.dot(normalized1, normalized2.T)
            
    def _save_cache(self):
        """Save embedding cache to disk"""
        try:
            with open(self._cache_file, 'wb') as f:
                pickle.dump(self._embedding_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
            
    def _load_cache(self):
        """Load embedding cache from disk"""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'rb') as f:
                    self._embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self._embedding_cache = {}
                
    def get_info(self) -> Dict:
        """Get information about the embedder"""
        return {
            "model": self.model_type,
            "mode": self._mode,
            "embedding_dim": self.model_config["dim"],
            "max_length": self.model_config["max_length"],
            "normalized": self.normalize,
            "cache_size": len(self._embedding_cache)
        }