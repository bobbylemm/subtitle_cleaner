from typing import List, Dict, Any, Optional
from app.domain.models import Segment, SubtitleDocument, Settings
from app.domain.constants import (
    MIN_SEGMENT_DURATION_MS,
    MAX_SEGMENT_DURATION_MS,
    MAX_CPS,
    MAX_LINE_LENGTH,
    MAX_LINES_PER_SEGMENT,
)


class ValidationIssue:
    """Represents a validation issue"""
    def __init__(
        self,
        segment_idx: Optional[int],
        issue_type: str,
        severity: str,
        message: str,
        **details
    ):
        self.segment_idx = segment_idx
        self.issue_type = issue_type
        self.severity = severity  # error, warning, info
        self.message = message
        self.details = details
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_idx": self.segment_idx,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


class SubtitleValidator:
    """Validate subtitle documents"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
    
    def validate(self, document: SubtitleDocument, strict: bool = False) -> List[ValidationIssue]:
        """Validate a subtitle document"""
        issues = []
        
        # Check for empty document
        if not document.segments:
            issues.append(ValidationIssue(
                None, "empty_document", "error",
                "Document contains no segments"
            ))
            return issues
        
        # Validate each segment
        for segment in document.segments:
            issues.extend(self._validate_segment(segment, strict))
        
        # Check for overlaps
        overlaps = document.find_overlaps()
        for idx1, idx2 in overlaps:
            issues.append(ValidationIssue(
                idx1, "overlap", "error",
                f"Segment {idx1} overlaps with segment {idx2}",
                overlapping_segments=[idx1, idx2]
            ))
        
        # Check chronological order
        issues.extend(self._validate_chronological_order(document))
        
        return issues
    
    def _validate_segment(self, segment: Segment, strict: bool) -> List[ValidationIssue]:
        """Validate a single segment"""
        issues = []
        
        # Check duration
        if segment.duration_ms < MIN_SEGMENT_DURATION_MS:
            issues.append(ValidationIssue(
                segment.idx, "duration_too_short", "error",
                f"Segment duration ({segment.duration_ms}ms) is below minimum ({MIN_SEGMENT_DURATION_MS}ms)",
                duration_ms=segment.duration_ms
            ))
        elif segment.duration_ms < self.settings.min_duration_ms:
            issues.append(ValidationIssue(
                segment.idx, "duration_short", "warning",
                f"Segment duration ({segment.duration_ms}ms) is below recommended ({self.settings.min_duration_ms}ms)",
                duration_ms=segment.duration_ms
            ))
        
        if segment.duration_ms > MAX_SEGMENT_DURATION_MS:
            issues.append(ValidationIssue(
                segment.idx, "duration_too_long", "error",
                f"Segment duration ({segment.duration_ms}ms) exceeds maximum ({MAX_SEGMENT_DURATION_MS}ms)",
                duration_ms=segment.duration_ms
            ))
        elif segment.duration_ms > self.settings.max_duration_ms:
            issues.append(ValidationIssue(
                segment.idx, "duration_long", "warning",
                f"Segment duration ({segment.duration_ms}ms) exceeds recommended ({self.settings.max_duration_ms}ms)",
                duration_ms=segment.duration_ms
            ))
        
        # Check CPS
        cps = segment.cps
        if cps > MAX_CPS:
            issues.append(ValidationIssue(
                segment.idx, "cps_too_high", "error",
                f"CPS ({cps:.1f}) exceeds maximum ({MAX_CPS})",
                cps=cps
            ))
        elif cps > self.settings.max_cps:
            issues.append(ValidationIssue(
                segment.idx, "cps_high", "warning",
                f"CPS ({cps:.1f}) exceeds target ({self.settings.max_cps})",
                cps=cps
            ))
        
        # Check text
        if not segment.text.strip():
            issues.append(ValidationIssue(
                segment.idx, "empty_text", "error",
                "Segment has empty text"
            ))
        
        # Check line length and count
        lines = segment.text.split('\n')
        if len(lines) > MAX_LINES_PER_SEGMENT:
            issues.append(ValidationIssue(
                segment.idx, "too_many_lines", "error" if strict else "warning",
                f"Segment has {len(lines)} lines (max: {MAX_LINES_PER_SEGMENT})",
                line_count=len(lines)
            ))
        
        for i, line in enumerate(lines):
            if len(line) > MAX_LINE_LENGTH:
                issues.append(ValidationIssue(
                    segment.idx, "line_too_long", "error" if strict else "warning",
                    f"Line {i+1} has {len(line)} characters (max: {MAX_LINE_LENGTH})",
                    line_number=i+1,
                    line_length=len(line)
                ))
            elif len(line) > self.settings.line_wrap:
                issues.append(ValidationIssue(
                    segment.idx, "line_long", "warning",
                    f"Line {i+1} has {len(line)} characters (target: {self.settings.line_wrap})",
                    line_number=i+1,
                    line_length=len(line)
                ))
        
        # Check timing
        if segment.start_ms < 0:
            issues.append(ValidationIssue(
                segment.idx, "negative_start", "error",
                "Segment has negative start time",
                start_ms=segment.start_ms
            ))
        
        if segment.end_ms <= segment.start_ms:
            issues.append(ValidationIssue(
                segment.idx, "invalid_timing", "error",
                "End time must be after start time",
                start_ms=segment.start_ms,
                end_ms=segment.end_ms
            ))
        
        return issues
    
    def _validate_chronological_order(self, document: SubtitleDocument) -> List[ValidationIssue]:
        """Check if segments are in chronological order"""
        issues = []
        
        for i in range(1, len(document.segments)):
            curr = document.segments[i]
            prev = document.segments[i-1]
            
            if curr.start_ms < prev.start_ms:
                issues.append(ValidationIssue(
                    curr.idx, "out_of_order", "error",
                    f"Segment {curr.idx} starts before segment {prev.idx}",
                    current_start=curr.start_ms,
                    previous_start=prev.start_ms
                ))
            
            # Check for gaps (optional)
            gap_ms = curr.start_ms - prev.end_ms
            if gap_ms > 10000:  # More than 10 seconds gap
                issues.append(ValidationIssue(
                    curr.idx, "large_gap", "warning",
                    f"Large gap ({gap_ms}ms) between segments {prev.idx} and {curr.idx}",
                    gap_ms=gap_ms
                ))
            elif gap_ms < -100:  # Overlap by more than 100ms
                # Already handled by overlap check, but could be more specific
                pass
        
        return issues
    
    @staticmethod
    def get_stats(document: SubtitleDocument) -> Dict[str, Any]:
        """Get statistics about the document"""
        if not document.segments:
            return {
                "segment_count": 0,
                "total_duration_ms": 0,
                "avg_duration_ms": 0,
                "avg_cps": 0,
                "max_cps": 0,
            }
        
        durations = [seg.duration_ms for seg in document.segments]
        cps_values = [seg.cps for seg in document.segments if seg.duration_ms > 0]
        
        return {
            "segment_count": len(document.segments),
            "total_duration_ms": document.duration_ms,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "avg_cps": sum(cps_values) / len(cps_values) if cps_values else 0,
            "max_cps": max(cps_values) if cps_values else 0,
            "total_characters": sum(len(seg.text) for seg in document.segments),
            "overlap_count": len(document.find_overlaps()),
        }