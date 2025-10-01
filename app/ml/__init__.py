"""ML-enhanced subtitle processing modules"""

from .base import MLModel, MLConfig, MLResult
from .gatekeeper import Gatekeeper, GatekeeperConfig, EditConstraints

# Optional imports (require torch and other ML dependencies)
try:
    from .punctuation import PunctuationModel
    from .grammar import GrammarCorrectionModel
    from .filler import ContextualFillerDetector
    from .scorer import PerplexityScorer
    
    __all__ = [
        "MLModel",
        "MLConfig",
        "MLResult",
        "Gatekeeper",
        "GatekeeperConfig",
        "EditConstraints",
        "PunctuationModel",
        "GrammarCorrectionModel",
        "ContextualFillerDetector",
        "PerplexityScorer",
    ]
except ImportError:
    # Minimal exports when ML dependencies not available
    __all__ = [
        "MLModel",
        "MLConfig",
        "MLResult",
        "Gatekeeper",
        "GatekeeperConfig",
        "EditConstraints",
    ]