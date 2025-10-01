from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class ProcessingMetrics(BaseModel):
    """Metrics from subtitle processing"""
    # Segment counts
    total_segments: int = Field(ge=0)
    segments_merged: int = Field(ge=0)
    segments_split: int = Field(ge=0)
    segments_removed: int = Field(ge=0)
    
    # CPS statistics
    avg_cps_before: float = Field(ge=0)
    avg_cps_after: float = Field(ge=0)
    max_cps_before: float = Field(ge=0)
    max_cps_after: float = Field(ge=0)
    cps_violations_fixed: int = Field(ge=0)
    
    # Timing fixes
    overlaps_fixed: int = Field(ge=0)
    duration_violations_fixed: int = Field(ge=0)
    
    # Text modifications
    fillers_removed: int = Field(ge=0)
    glossary_replacements: int = Field(ge=0)
    lines_wrapped: int = Field(ge=0)
    punctuation_normalized: int = Field(ge=0)
    
    # Performance
    processing_time_ms: int = Field(ge=0)
    file_size_before: int = Field(ge=0)
    file_size_after: int = Field(ge=0)


class ValidationIssue(BaseModel):
    """Single validation issue"""
    segment_index: Optional[int] = None
    line_number: Optional[int] = None
    issue_type: str
    severity: str = Field(pattern="^(error|warning|info)$")
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ProcessingReport(BaseModel):
    """Detailed processing report"""
    job_id: str
    timestamp: datetime
    
    # Input information
    input_format: str
    input_language: str
    input_segments: int
    input_duration_ms: int
    
    # Processing details
    settings_used: Dict[str, Any]
    steps_performed: List[str]
    
    # Metrics
    metrics: ProcessingMetrics
    
    # Issues and violations
    validation_issues: List[ValidationIssue] = Field(default_factory=list)
    violations: Dict[str, int] = Field(default_factory=dict)
    
    # Modifications log
    modifications: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    
    # Summary
    success: bool
    error_message: Optional[str] = None


class ReportSummary(BaseModel):
    """Simplified report summary"""
    job_id: str
    success: bool
    segments_processed: int
    improvements: Dict[str, Any] = Field(default_factory=dict)
    key_metrics: Dict[str, float] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)


class BatchReport(BaseModel):
    """Report for batch processing"""
    batch_id: str
    timestamp: datetime
    total_files: int
    successful: int
    failed: int
    reports: List[ReportSummary]
    aggregated_metrics: ProcessingMetrics
    total_processing_time_ms: int