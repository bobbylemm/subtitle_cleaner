"""
Integration module for the new scalable correction pipeline.
Manages model lifecycle and provides clean API interface.
"""

import logging
from pathlib import Path
from typing import Optional
from functools import lru_cache

from app.services.subtitle_correction_pipeline import (
    SubtitleCorrectionPipeline,
    ModelManager
)

logger = logging.getLogger(__name__)

# Global model manager instance
_model_manager: Optional[ModelManager] = None
_pipeline_instance: Optional[SubtitleCorrectionPipeline] = None


def initialize_models():
    """Initialize all models at app startup.
    This should be called once during app initialization.
    """
    global _model_manager
    
    logger.info("Initializing model manager and preloading models...")
    
    _model_manager = ModelManager()
    
    # Preload commonly used models
    try:
        # Embedding model
        from sentence_transformers import SentenceTransformer
        _model_manager.get_model(
            "embedder",
            lambda: SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        )
        
        # MLM model
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        import torch
        
        def load_mlm():
            model_name = "bert-base-multilingual-cased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            model.eval()
            
            if torch.cuda.is_available():
                model = model.cuda()
                logger.info("MLM model loaded on GPU")
            
            return {"tokenizer": tokenizer, "model": model}
        
        _model_manager.get_model("mlm", load_mlm)
        
        # NER model
        try:
            import spacy
            
            def load_ner():
                try:
                    # Try multilingual first
                    return spacy.load("xx_ent_wiki_sm")
                except:
                    # Fallback to English
                    return spacy.load("en_core_web_sm")
            
            _model_manager.get_model("ner", load_ner)
        except Exception as e:
            logger.warning(f"NER model loading failed: {e}")
        
        # SymSpell
        try:
            from symspellpy import SymSpell
            
            def load_symspell():
                sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                dict_path = Path(__file__).parent.parent / "resources" / "frequency_dictionary_en_82_765.txt"
                
                if dict_path.exists():
                    sym_spell.load_dictionary(str(dict_path), term_index=0, count_index=1)
                    logger.info("SymSpell dictionary loaded")
                
                return sym_spell
            
            _model_manager.get_model("symspell", load_symspell)
        except Exception as e:
            logger.warning(f"SymSpell loading failed: {e}")
        
        logger.info("Model initialization complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise


@lru_cache(maxsize=1)
def get_pipeline(config_path: Optional[str] = None) -> SubtitleCorrectionPipeline:
    """Get or create the correction pipeline instance.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        SubtitleCorrectionPipeline instance
    """
    global _pipeline_instance
    
    if _pipeline_instance is None:
        # Use default config path if not provided
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config" / "correction_pipeline.yaml")
        
        logger.info(f"Creating pipeline with config: {config_path}")
        _pipeline_instance = SubtitleCorrectionPipeline(config_path)
        
        # Inject preloaded models
        if _model_manager:
            # Update stages with preloaded models
            if "embedder" in _model_manager._models:
                _pipeline_instance.context_builder._embedder = _model_manager._models["embedder"]
            
            if "ner" in _model_manager._models:
                _pipeline_instance.guard_system._ner = _model_manager._models["ner"]
            
            if "symspell" in _model_manager._models:
                _pipeline_instance.pass1._symspell = _model_manager._models["symspell"]
    
    return _pipeline_instance


def correct_subtitles_scalable(srt_content: str, 
                               enable_robust: bool = True,
                               custom_config: Optional[dict] = None) -> str:
    """
    Main entry point for scalable subtitle correction.
    
    Args:
        srt_content: SRT format subtitle content
        enable_robust: Whether to use the robust pipeline (vs simple)
        custom_config: Optional config overrides
        
    Returns:
        Corrected SRT content
    """
    if not enable_robust:
        # Use simple correction for backwards compatibility
        return srt_content
    
    # Get pipeline instance
    pipeline = get_pipeline()
    
    # Apply custom config if provided
    if custom_config:
        original_config = pipeline.config.copy()
        pipeline.config.update(custom_config)
        
        try:
            # Process with custom config
            result = pipeline.process(srt_content)
        finally:
            # Restore original config
            pipeline.config = original_config
    else:
        # Process with default config
        result = pipeline.process(srt_content)
    
    return result


def get_model_manager() -> ModelManager:
    """Get the model manager instance."""
    global _model_manager
    
    if _model_manager is None:
        initialize_models()
    
    return _model_manager


# Health check for models
def check_models_health() -> dict:
    """Check if all models are loaded and healthy."""
    health = {
        "status": "healthy",
        "models": {}
    }
    
    if _model_manager:
        for model_name in ["embedder", "mlm", "ner", "symspell"]:
            health["models"][model_name] = model_name in _model_manager._models
    else:
        health["status"] = "not_initialized"
    
    if _pipeline_instance:
        health["pipeline"] = "ready"
    else:
        health["pipeline"] = "not_initialized"
    
    # Overall status
    if health.get("pipeline") != "ready":
        health["status"] = "degraded"
    
    return health


# Cleanup function for graceful shutdown
def cleanup_models():
    """Clean up models and free memory."""
    global _model_manager, _pipeline_instance
    
    logger.info("Cleaning up models...")
    
    if _model_manager:
        _model_manager._models.clear()
        _model_manager = None
    
    _pipeline_instance = None
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear GPU memory if using CUDA
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    logger.info("Model cleanup complete")