"""Change tracking system for subtitle modifications with detailed reporting"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from deepdiff import DeepDiff
import hashlib
from datetime import datetime


class ChangeType(str, Enum):
    """Types of changes that can be made to subtitles"""
    TEXT_MODIFICATION = "text_modification"
    PUNCTUATION_RESTORED = "punctuation_restored"
    TRUECASING_APPLIED = "truecasing_applied"
    GRAMMAR_CORRECTED = "grammar_corrected"
    FILLER_REMOVED = "filler_removed"
    ENTITY_STABILIZED = "entity_stabilized"
    TIMING_ADJUSTED = "timing_adjusted"
    SEGMENT_MERGED = "segment_merged"
    SEGMENT_SPLIT = "segment_split"
    LINE_WRAPPED = "line_wrapped"
    CPS_OPTIMIZED = "cps_optimized"
    OVERLAP_FIXED = "overlap_fixed"
    GLOSSARY_APPLIED = "glossary_applied"


class ChangeSource(str, Enum):
    """Source/model that made the change"""
    RULE_BASED = "rule_based"
    ONNX_PUNCT = "onnx_punctuation"
    GECTOR = "gector"
    LANGUAGE_TOOL = "language_tool"
    PHONETIC_MATCH = "phonetic_match"
    USER_CONTEXT = "user_context"
    USER_GLOSSARY = "user_glossary"
    MAJORITY_VOTE = "majority_vote"


@dataclass
class TextDiff:
    """Detailed text difference information"""
    start_pos: int
    end_pos: int
    old_text: str
    new_text: str
    edit_distance: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Change:
    """Represents a single change made to the subtitle"""
    segment_idx: int
    change_type: ChangeType
    field: str  # "text", "start_ms", "end_ms", etc.
    old_value: Any
    new_value: Any
    reason: str
    confidence: float = 1.0
    source: ChangeSource = ChangeSource.RULE_BASED
    text_diff: Optional[TextDiff] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = {
            "segment_idx": self.segment_idx,
            "change_type": self.change_type.value,
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "confidence": self.confidence,
            "source": self.source.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
        if self.text_diff:
            result["text_diff"] = self.text_diff.to_dict()
        return result


class ChangeTracker:
    """Tracks all changes made during subtitle processing"""
    
    def __init__(self):
        self.changes: List[Change] = []
        self._original_hash: Optional[str] = None
        self._final_hash: Optional[str] = None
        self._processing_time_ms: int = 0
        
    def set_original_document(self, segments: List[Dict]) -> None:
        """Store hash of original document for verification"""
        doc_str = json.dumps(segments, sort_keys=True)
        self._original_hash = hashlib.sha256(doc_str.encode()).hexdigest()[:16]
        
    def set_final_document(self, segments: List[Dict]) -> None:
        """Store hash of final document"""
        doc_str = json.dumps(segments, sort_keys=True)
        self._final_hash = hashlib.sha256(doc_str.encode()).hexdigest()[:16]
        
    def track_text_change(
        self,
        segment_idx: int,
        old_text: str,
        new_text: str,
        change_type: ChangeType,
        reason: str,
        source: ChangeSource = ChangeSource.RULE_BASED,
        confidence: float = 1.0,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Track a text modification with detailed diff"""
        
        # Skip if no actual change
        if old_text == new_text:
            return
            
        # Calculate text diff details
        text_diff = self._calculate_text_diff(old_text, new_text)
        
        change = Change(
            segment_idx=segment_idx,
            change_type=change_type,
            field="text",
            old_value=old_text,
            new_value=new_text,
            reason=reason,
            confidence=confidence,
            source=source,
            text_diff=text_diff,
            metadata=metadata or {}
        )
        
        self.changes.append(change)
        
    def track_timing_change(
        self,
        segment_idx: int,
        field: str,  # "start_ms" or "end_ms"
        old_value: int,
        new_value: int,
        reason: str
    ) -> None:
        """Track timing adjustments"""
        
        if old_value == new_value:
            return
            
        change = Change(
            segment_idx=segment_idx,
            change_type=ChangeType.TIMING_ADJUSTED,
            field=field,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            metadata={
                "delta_ms": new_value - old_value,
                "delta_seconds": (new_value - old_value) / 1000.0
            }
        )
        
        self.changes.append(change)
        
    def track_segment_merge(
        self,
        segments: List[int],
        reason: str
    ) -> None:
        """Track when segments are merged"""
        
        change = Change(
            segment_idx=segments[0],  # First segment index
            change_type=ChangeType.SEGMENT_MERGED,
            field="segments",
            old_value=segments,
            new_value=segments[0],
            reason=reason,
            metadata={
                "merged_count": len(segments),
                "merged_indices": segments
            }
        )
        
        self.changes.append(change)
        
    def _calculate_text_diff(self, old_text: str, new_text: str) -> TextDiff:
        """Calculate detailed text difference"""
        
        # Find the common prefix
        prefix_len = 0
        for i, (c1, c2) in enumerate(zip(old_text, new_text)):
            if c1 != c2:
                break
            prefix_len = i + 1
            
        # Find the common suffix
        suffix_len = 0
        for i, (c1, c2) in enumerate(zip(reversed(old_text), reversed(new_text))):
            if c1 != c2:
                break
            suffix_len = i + 1
            
        # Extract the different parts
        old_diff = old_text[prefix_len:len(old_text) - suffix_len if suffix_len else None]
        new_diff = new_text[prefix_len:len(new_text) - suffix_len if suffix_len else None]
        
        # Calculate edit distance
        edit_distance = self._levenshtein_distance(old_text, new_text)
        
        return TextDiff(
            start_pos=prefix_len,
            end_pos=len(old_text) - suffix_len,
            old_text=old_diff,
            new_text=new_diff,
            edit_distance=edit_distance
        )
        
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row are one character longer
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive change report"""
        
        # Group changes by type
        changes_by_type = {}
        for change in self.changes:
            change_type = change.change_type.value
            if change_type not in changes_by_type:
                changes_by_type[change_type] = []
            changes_by_type[change_type].append(change.to_dict())
            
        # Calculate statistics
        stats = {
            "total_changes": len(self.changes),
            "segments_modified": len(set(c.segment_idx for c in self.changes)),
            "changes_by_type": {k: len(v) for k, v in changes_by_type.items()},
            "average_confidence": (
                sum(c.confidence for c in self.changes) / len(self.changes)
                if self.changes else 1.0
            ),
            "total_edit_distance": sum(
                c.text_diff.edit_distance 
                for c in self.changes 
                if c.text_diff
            )
        }
        
        # Build report
        report = {
            "summary": {
                "original_hash": self._original_hash,
                "final_hash": self._final_hash,
                "processing_time_ms": self._processing_time_ms,
                **stats
            },
            "changes": [c.to_dict() for c in self.changes],
            "changes_by_type": changes_by_type,
            "changes_by_segment": self._group_by_segment()
        }
        
        return report
        
    def _group_by_segment(self) -> Dict[int, List[Dict]]:
        """Group changes by segment index"""
        by_segment = {}
        for change in self.changes:
            idx = change.segment_idx
            if idx not in by_segment:
                by_segment[idx] = []
            by_segment[idx].append(change.to_dict())
        return by_segment
        
    def get_confidence_threshold_changes(self, threshold: float = 0.8) -> List[Change]:
        """Get only changes above confidence threshold"""
        return [c for c in self.changes if c.confidence >= threshold]
        
    def get_text_changes_summary(self) -> str:
        """Generate human-readable summary of text changes"""
        lines = []
        
        for change in self.changes:
            if change.field == "text" and change.text_diff:
                lines.append(
                    f"Segment {change.segment_idx}: "
                    f"{change.change_type.value} - "
                    f"'{change.text_diff.old_text}' → '{change.text_diff.new_text}' "
                    f"({change.reason})"
                )
                
        return "\n".join(lines)