"""Gatekeeper system for ML model outputs"""

from dataclasses import dataclass
from typing import List, Set, Dict, Any, Optional, Tuple
import re
import difflib
import logging

from app.ml.base import MLResult, EditType
from app.domain.constants import Language

logger = logging.getLogger(__name__)


@dataclass
class EditConstraints:
    """Constraints for ML edits"""
    max_edit_ratio: float = 0.15  # Max 15% of tokens can change
    max_char_change: int = 8  # Max 8 character net change per segment
    allowed_edit_types: Set[EditType] = None
    protected_entities: List[str] = None  # NER/glossary terms to protect
    protected_patterns: List[re.Pattern] = None  # Regex patterns to protect
    require_perplexity_improvement: bool = True
    min_confidence: float = 0.7
    
    def __post_init__(self):
        if self.allowed_edit_types is None:
            # Default allowed edits
            self.allowed_edit_types = {
                EditType.INSERT_PUNCTUATION,
                EditType.REMOVE_PUNCTUATION,
                EditType.CHANGE_CASE,
                EditType.INSERT_ARTICLE,
                EditType.REMOVE_ARTICLE,
                EditType.FIX_AGREEMENT,
                EditType.FIX_APOSTROPHE,
                EditType.REMOVE_FILLER,
                EditType.FIX_SPACING,
                EditType.FIX_DIACRITIC,
            }
        
        if self.protected_patterns is None:
            # Default patterns to protect
            self.protected_patterns = [
                re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'),  # Proper nouns
                re.compile(r'\b\d+(?:\.\d+)?\b'),  # Numbers
                re.compile(r'\b[A-Z]{2,}\b'),  # Acronyms
                re.compile(r'https?://[^\s]+'),  # URLs
                re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),  # Emails
            ]


@dataclass
class GatekeeperConfig:
    """Configuration for the gatekeeper"""
    language: Language
    constraints: EditConstraints
    strict_mode: bool = False  # If True, reject any violation
    log_violations: bool = True
    preserve_glossary: bool = True
    preserve_entities: bool = True


class Gatekeeper:
    """Validates and constrains ML model outputs"""
    
    def __init__(self, config: GatekeeperConfig):
        self.config = config
        self.violations: List[Dict[str, Any]] = []
    
    def validate(
        self,
        ml_result: MLResult,
        perplexity_before: Optional[float] = None,
        perplexity_after: Optional[float] = None,
        glossary_terms: Optional[Set[str]] = None,
        entities: Optional[List[str]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate ML model output against constraints
        
        Returns:
            - is_valid: Whether the output passes all checks
            - text_to_use: Either predicted or original text
            - validation_report: Details of validation checks
        """
        self.violations = []
        report = {
            "checks_passed": [],
            "checks_failed": [],
            "edit_stats": {},
            "decision": None
        }
        
        # 1. Check edit ratio
        if not self._check_edit_ratio(ml_result):
            report["checks_failed"].append("edit_ratio_exceeded")
            if self.config.strict_mode:
                return self._reject(ml_result, report, "Edit ratio exceeded")
        else:
            report["checks_passed"].append("edit_ratio_ok")
        
        # 2. Check character change
        if not self._check_char_change(ml_result):
            report["checks_failed"].append("char_change_exceeded")
            if self.config.strict_mode:
                return self._reject(ml_result, report, "Character change exceeded")
        else:
            report["checks_passed"].append("char_change_ok")
        
        # 3. Check edit types
        if not self._check_edit_types(ml_result):
            report["checks_failed"].append("invalid_edit_types")
            if self.config.strict_mode:
                return self._reject(ml_result, report, "Invalid edit types")
        else:
            report["checks_passed"].append("edit_types_ok")
        
        # 4. Check protected entities
        if self.config.preserve_entities and entities:
            if not self._check_entities_preserved(ml_result, entities):
                report["checks_failed"].append("entities_modified")
                return self._reject(ml_result, report, "Protected entities modified")
        else:
            report["checks_passed"].append("entities_preserved")
        
        # 5. Check glossary terms
        if self.config.preserve_glossary and glossary_terms:
            if not self._check_glossary_preserved(ml_result, glossary_terms):
                report["checks_failed"].append("glossary_modified")
                return self._reject(ml_result, report, "Glossary terms modified")
        else:
            report["checks_passed"].append("glossary_preserved")
        
        # 6. Check protected patterns
        if not self._check_protected_patterns(ml_result):
            report["checks_failed"].append("protected_patterns_modified")
            if self.config.strict_mode:
                return self._reject(ml_result, report, "Protected patterns modified")
        else:
            report["checks_passed"].append("patterns_preserved")
        
        # 7. Check perplexity improvement
        if self.config.constraints.require_perplexity_improvement:
            if not self._check_perplexity_improvement(perplexity_before, perplexity_after):
                report["checks_failed"].append("perplexity_worse")
                if self.config.strict_mode:
                    return self._reject(ml_result, report, "Perplexity did not improve")
        else:
            report["checks_passed"].append("perplexity_improved")
        
        # 8. Check confidence
        if ml_result.confidence < self.config.constraints.min_confidence:
            report["checks_failed"].append("low_confidence")
            if self.config.strict_mode:
                return self._reject(ml_result, report, f"Low confidence: {ml_result.confidence}")
        else:
            report["checks_passed"].append("confidence_ok")
        
        # Accept the ML output
        report["decision"] = "accepted"
        report["edit_stats"] = {
            "edit_distance": ml_result.edit_distance,
            "edit_ratio": ml_result.edit_ratio,
            "confidence": ml_result.confidence,
            "num_edits": len(ml_result.edits)
        }
        
        if self.config.log_violations and self.violations:
            logger.warning(f"Gatekeeper violations (non-strict): {self.violations}")
        
        return True, ml_result.predicted_text, report
    
    def _reject(self, ml_result: MLResult, report: Dict, reason: str) -> Tuple[bool, str, Dict]:
        """Reject ML output and return original"""
        report["decision"] = f"rejected: {reason}"
        if self.config.log_violations:
            logger.info(f"Gatekeeper rejected ML output: {reason}")
        return False, ml_result.original_text, report
    
    def _check_edit_ratio(self, result: MLResult) -> bool:
        """Check if edit ratio is within bounds"""
        return result.edit_ratio <= self.config.constraints.max_edit_ratio
    
    def _check_char_change(self, result: MLResult) -> bool:
        """Check if character change is within bounds"""
        char_diff = abs(len(result.predicted_text) - len(result.original_text))
        return char_diff <= self.config.constraints.max_char_change
    
    def _check_edit_types(self, result: MLResult) -> bool:
        """Check if all edits are allowed types"""
        for edit_type in result.edit_types:
            if edit_type not in self.config.constraints.allowed_edit_types:
                self.violations.append({
                    "type": "invalid_edit_type",
                    "edit_type": edit_type
                })
                return False
        return True
    
    def _check_entities_preserved(self, result: MLResult, entities: List[str]) -> bool:
        """Check if named entities are preserved"""
        original_lower = result.original_text.lower()
        predicted_lower = result.predicted_text.lower()
        
        for entity in entities:
            entity_lower = entity.lower()
            # Check if entity appears same number of times
            if original_lower.count(entity_lower) != predicted_lower.count(entity_lower):
                self.violations.append({
                    "type": "entity_modified",
                    "entity": entity
                })
                return False
        return True
    
    def _check_glossary_preserved(self, result: MLResult, glossary_terms: Set[str]) -> bool:
        """Check if glossary terms are preserved"""
        for term in glossary_terms:
            if term in result.original_text:
                if term not in result.predicted_text:
                    self.violations.append({
                        "type": "glossary_modified",
                        "term": term
                    })
                    return False
        return True
    
    def _check_protected_patterns(self, result: MLResult) -> bool:
        """Check if protected patterns are preserved"""
        for pattern in self.config.constraints.protected_patterns:
            original_matches = set(pattern.findall(result.original_text))
            predicted_matches = set(pattern.findall(result.predicted_text))
            
            if original_matches != predicted_matches:
                self.violations.append({
                    "type": "pattern_modified",
                    "pattern": pattern.pattern,
                    "original_matches": list(original_matches),
                    "predicted_matches": list(predicted_matches)
                })
                if self.config.strict_mode:
                    return False
        
        return len(self.violations) == 0 or not self.config.strict_mode
    
    def _check_perplexity_improvement(
        self,
        before: Optional[float],
        after: Optional[float]
    ) -> bool:
        """Check if perplexity improved (lower is better)"""
        if before is None or after is None:
            return True  # Skip check if no scores
        return after <= before
    
    def mask_protected_content(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        glossary_terms: Optional[Set[str]] = None
    ) -> Tuple[str, Dict[str, str]]:
        """
        Mask protected content before ML processing
        Returns masked text and mapping for unmasking
        """
        mask_map = {}
        mask_counter = 0
        masked_text = text
        
        # Mask entities
        if entities:
            for entity in sorted(entities, key=len, reverse=True):
                if entity in masked_text:
                    mask_key = f"⟦ENT_{mask_counter}⟧"
                    masked_text = masked_text.replace(entity, mask_key)
                    mask_map[mask_key] = entity
                    mask_counter += 1
        
        # Mask glossary terms
        if glossary_terms:
            for term in sorted(glossary_terms, key=len, reverse=True):
                if term in masked_text:
                    mask_key = f"⟦GLOSS_{mask_counter}⟧"
                    masked_text = masked_text.replace(term, mask_key)
                    mask_map[mask_key] = term
                    mask_counter += 1
        
        # Mask patterns (numbers, URLs, etc.)
        for pattern in self.config.constraints.protected_patterns:
            for match in pattern.finditer(masked_text):
                mask_key = f"⟦PAT_{mask_counter}⟧"
                masked_text = masked_text[:match.start()] + mask_key + masked_text[match.end():]
                mask_map[mask_key] = match.group()
                mask_counter += 1
        
        return masked_text, mask_map
    
    def unmask_content(self, text: str, mask_map: Dict[str, str]) -> str:
        """Unmask previously masked content"""
        result = text
        for mask_key, original in mask_map.items():
            result = result.replace(mask_key, original)
        return result