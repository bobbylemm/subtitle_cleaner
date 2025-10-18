from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import List, Optional, Dict, Any, Annotated
from typing_extensions import Self

from app.domain.constants import SubtitleFormat, Language

# Custom types using Annotated pattern (Pydantic v2 best practice)
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonEmptyStr = Annotated[str, Field(min_length=1)]


class Segment(BaseModel):
    """API representation of a subtitle segment"""
    model_config = ConfigDict(
        str_strip_whitespace=True,  # Auto-strip strings
        validate_assignment=True,   # Validate on field assignment
        json_schema_extra={
            "example": {
                "start_ms": 1000,
                "end_ms": 4000,
                "text": "Hello, world!"
            }
        }
    )
    
    start_ms: NonNegativeInt = Field(description="Start time in milliseconds")
    end_ms: PositiveInt = Field(description="End time in milliseconds")
    text: NonEmptyStr = Field(description="Segment text")
    
    @field_validator("text", mode="after")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure text is not empty after stripping"""
        if not v:
            raise ValueError("Text cannot be empty")
        return v
    
    @model_validator(mode="after")
    def validate_timing(self) -> Self:
        """Ensure end time is after start time"""
        if self.end_ms <= self.start_ms:
            raise ValueError(f"End time ({self.end_ms}ms) must be after start time ({self.start_ms}ms)")
        return self


class SubtitleDocument(BaseModel):
    """API representation of a complete subtitle document"""
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=False,
        json_schema_extra={
            "example": {
                "segments": [
                    {"start_ms": 0, "end_ms": 2000, "text": "First segment"},
                    {"start_ms": 2100, "end_ms": 4000, "text": "Second segment"}
                ],
                "format": "srt",
                "language": "en"
            }
        }
    )
    
    segments: Annotated[List[Segment], Field(min_length=1, description="List of subtitle segments")]
    format: SubtitleFormat = Field(default=SubtitleFormat.SRT, description="Subtitle format")
    language: Language = Field(default=Language.EN, description="Content language")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator("segments", mode="after")
    @classmethod
    def validate_segments(cls, v: List[Segment]) -> List[Segment]:
        """Validate segment ordering and consistency"""
        if not v:
            raise ValueError("Document must have at least one segment")
        
        # Check for chronological order
        for i in range(1, len(v)):
            if v[i].start_ms < v[i-1].start_ms:
                raise ValueError(
                    f"Segments must be in chronological order. "
                    f"Segment {i} starts at {v[i].start_ms}ms but segment {i-1} starts at {v[i-1].start_ms}ms"
                )
        
        return v


class CleanRequest(BaseModel):
    """Request to clean subtitles"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "content": "1\n00:00:01,000 --> 00:00:03,000\nHello world\n",
                "format": "srt",
                "language": "en",
                "settings": {"max_cps": 17.0, "merge_mode": "smart"}
            }
        }
    )
    
    content: Annotated[
        str, 
        Field(
            min_length=1,
            max_length=10 * 1024 * 1024,  # 10MB limit
            description="Raw subtitle content (can be base64 encoded)"
        )
    ]
    is_base64: bool = Field(default=False, description="Whether content is base64 encoded")
    format: SubtitleFormat = Field(default=SubtitleFormat.SRT, description="Input format")
    language: Language = Field(default=Language.EN, description="Content language")
    settings: Optional[Dict[str, Any]] = Field(default=None, description="Processing settings")
    
    # Enhanced features (Layers 3-6)
    context_sources: Optional[List[Dict[str, Any]]] = Field(default=None, description="Context sources for Layer 3")
    enable_retrieval: Optional[bool] = Field(default=False, description="Enable Layer 4 retrieval")
    enable_llm: Optional[bool] = Field(default=False, description="Enable Layer 5 LLM selection")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key for Layer 5")
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID for Layer 6")
    region_hint: Optional[str] = Field(default=None, description="Regional hint for retrieval")
    
    # Auto-context generation
    context_mode: Optional[str] = Field(
        default="none",
        description="Context generation mode: 'none' (default), 'auto' (automatic), 'manual' (user-provided only), 'hybrid' (both), 'smart' (adaptive)",
        pattern="^(none|auto|manual|hybrid|smart)$"
    )
    
    # Contextual correction mode
    correction_mode: Optional[str] = Field(
        default="balanced",
        description="Contextual correction mode: 'legacy' (original behavior), 'conservative' (high precision), 'balanced' (balanced precision/recall), 'aggressive' (high recall)",
        pattern="^(legacy|conservative|balanced|aggressive)$"
    )
    
    # ML-based correction
    enable_ml_correction: Optional[bool] = Field(
        default=True,
        description="Enable ML-based context correction using BERT/T5 models"
    )
    
    ml_correction_mode: Optional[str] = Field(
        default="balanced",
        description="ML correction mode: 'fast' (high confidence only), 'balanced' (medium), 'quality' (with LLM fallback)",
        pattern="^(fast|balanced|quality)$"
    )
    
    # Holistic correction (document-level understanding)
    enable_holistic_correction: Optional[bool] = Field(
        default=False,
        description="Enable holistic document-level context correction (understands entire document before corrections)"
    )
    
    @field_validator("content", mode="after")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensure content is not empty after stripping"""
        if not v:
            raise ValueError("Content cannot be empty")
        return v


class CleanResponse(BaseModel):
    """Response from subtitle cleaning"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "content": "1\n00:00:01,000 --> 00:00:03,000\nHello world\n",
                "format": "srt",
                "segments_processed": 10,
                "segments_modified": 3,
                "processing_time_ms": 150,
                "report": {"modifications": {"summary": {"merges": 2}}},
                "errors": []
            }
        }
    )
    
    success: bool = Field(description="Whether processing succeeded")
    content: Optional[str] = Field(default=None, description="Cleaned subtitle content (plain text)")
    cleaned_content: Optional[str] = Field(default=None, description="Cleaned subtitle content (base64 if input was base64)")
    format: SubtitleFormat = Field(description="Output format")
    segments_processed: NonNegativeInt = Field(description="Total segments processed")
    segments_modified: NonNegativeInt = Field(description="Segments that were modified")
    processing_time_ms: NonNegativeInt = Field(description="Processing time in milliseconds")
    report: Optional[Dict[str, Any]] = Field(default=None, description="Detailed processing report")
    errors: List[str] = Field(default_factory=list, description="Error messages if any")
    
    # Phase 2 and 3 fields removed - keeping only Layer 1-2 features


class ValidateRequest(BaseModel):
    """Request to validate subtitles"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    content: Annotated[str, Field(min_length=1, max_length=10 * 1024 * 1024, description="Raw subtitle content")]
    format: SubtitleFormat = Field(default=SubtitleFormat.SRT, description="Subtitle format")
    language: Language = Field(default=Language.EN, description="Content language")
    strict: bool = Field(default=False, description="Use strict validation rules")


class ValidateResponse(BaseModel):
    """Response from subtitle validation"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "valid": True,
                "format": "srt",
                "language": "en",
                "errors": [],
                "warnings": [{"issue_type": "cps_high", "severity": "warning", "message": "CPS too high"}],
                "stats": {"segment_count": 10, "avg_cps": 15.5}
            }
        }
    )
    
    valid: bool = Field(description="Whether the subtitle is valid")
    format: SubtitleFormat = Field(description="Detected format")
    language: Language = Field(description="Content language")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Validation errors")
    warnings: List[Dict[str, Any]] = Field(default_factory=list, description="Validation warnings")
    stats: Dict[str, Any] = Field(default_factory=dict, description="Document statistics")


class PreviewRequest(BaseModel):
    """Request to preview cleaning"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    content: Annotated[str, Field(min_length=1, max_length=10 * 1024 * 1024, description="Raw subtitle content")]
    format: SubtitleFormat = Field(default=SubtitleFormat.SRT, description="Subtitle format")
    language: Language = Field(default=Language.EN, description="Content language")
    settings: Optional[Dict[str, Any]] = Field(default=None, description="Processing settings")
    segment_indices: Optional[List[PositiveInt]] = Field(default=None, description="Preview specific segments")
    max_segments: Annotated[int, Field(ge=1, le=50)] = Field(default=10, description="Maximum segments to preview")


class PreviewResponse(BaseModel):
    """Response from preview"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "original": [{"start_ms": 0, "end_ms": 2000, "text": "Um, hello"}],
                "cleaned": [{"start_ms": 0, "end_ms": 2000, "text": "Hello"}],
                "changes": [{"type": "filler_removed", "segment_index": 0}],
                "estimated_reduction_percent": 15.5
            }
        }
    )
    
    original: List[Segment] = Field(description="Original segments")
    cleaned: List[Segment] = Field(description="Cleaned segments")
    changes: List[Dict[str, Any]] = Field(default_factory=list, description="List of changes made")
    estimated_reduction_percent: Annotated[float, Field(ge=0, le=100)] = Field(
        description="Estimated text reduction percentage"
    )