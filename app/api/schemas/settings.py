from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Annotated, Dict
from typing_extensions import Self

from app.domain.constants import MergeMode, FillerMode, Language


class ProcessingSettings(BaseModel):
    """Settings for subtitle processing"""
    # Merge settings
    merge_mode: MergeMode = Field(
        default=MergeMode.SMART,
        description="How aggressively to merge segments"
    )
    
    # Line wrapping
    line_wrap: int = Field(
        default=42,
        ge=20,
        le=80,
        description="Maximum characters per line"
    )
    max_lines: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Maximum lines per segment"
    )
    
    # Timing constraints
    min_duration_ms: int = Field(
        default=1500,
        ge=500,
        le=5000,
        description="Minimum segment duration in milliseconds"
    )
    max_duration_ms: int = Field(
        default=6000,
        ge=2000,
        le=10000,
        description="Maximum segment duration in milliseconds"
    )
    
    # Reading speed
    max_cps: float = Field(
        default=17.0,
        ge=5.0,
        le=25.0,
        description="Maximum characters per second"
    )
    
    # Filler handling
    filler_mode: FillerMode = Field(
        default=FillerMode.REMOVE,
        description="How to handle filler words"
    )
    custom_fillers: List[str] = Field(
        default_factory=list,
        description="Additional filler words to remove"
    )
    
    # Language and glossary
    language: Language = Field(
        default=Language.EN,
        description="Subtitle language"
    )
    glossary: Optional[str] = Field(
        default=None,
        description="Glossary ID to enforce"
    )
    
    # Additional options
    normalize_punctuation: bool = Field(
        default=True,
        description="Normalize punctuation (... to &, etc.)"
    )
    fix_overlaps: bool = Field(
        default=True,
        description="Fix overlapping segments"
    )
    remove_empty: bool = Field(
        default=True,
        description="Remove empty segments"
    )
    preserve_formatting: bool = Field(
        default=False,
        description="Preserve HTML/formatting tags"
    )
    
    # ML enhancement settings
    ml_enabled: bool = Field(
        default=False,
        description="Enable ML-based enhancements"
    )
    ml_models: Dict[str, bool] = Field(
        default_factory=lambda: {
            "punctuation": True,
            "grammar": True,
            "contextual_fillers": True,
            "scoring": True
        },
        description="Which ML models to enable"
    )
    ml_constraints: Dict[str, float] = Field(
        default_factory=lambda: {
            "max_edit_ratio": 0.15,
            "max_char_change": 8,
            "min_confidence": 0.7
        },
        description="Constraints for ML edits"
    )
    ml_device: str = Field(
        default="cpu",
        pattern="^(cpu|cuda)$",
        description="Device for ML inference (cpu or cuda)"
    )
    ml_quantized: bool = Field(
        default=True,
        description="Use quantized models for faster inference"
    )
    
    # Enhanced cleaning features
    enable_punctuation: bool = Field(
        default=False,
        description="Enable ONNX-based punctuation restoration and truecasing"
    )
    enable_entity_stabilization: bool = Field(
        default=False,
        description="Enable entity name stabilization using phonetic matching"
    )
    enable_context_extraction: bool = Field(
        default=False,
        description="Enable extraction of entities from user-provided context sources"
    )
    
    # Phase 2: Contextual Understanding features
    enable_topic_segmentation: bool = Field(
        default=False,
        description="Enable topic segmentation and coherence scoring"
    )
    enable_speaker_tracking: bool = Field(
        default=False,
        description="Enable speaker diarization from text patterns"
    )
    enable_coreference_resolution: bool = Field(
        default=False,
        description="Enable pronoun and entity coreference resolution"
    )
    
    # Phase 3: Advanced Processing features
    enable_domain_classification: bool = Field(
        default=False,
        description="Enable domain-specific processing (technical, medical, legal, etc.)"
    )
    enable_quality_scoring: bool = Field(
        default=False,
        description="Enable quality scoring and reporting"
    )
    enable_cps_optimization: bool = Field(
        default=False,
        description="Enable CPS (characters per second) optimization"
    )
    enable_vocabulary_enforcement: bool = Field(
        default=False,
        description="Enable domain vocabulary and glossary enforcement"
    )
    enable_adaptive_processing: bool = Field(
        default=False,
        description="Enable adaptive processing based on content analysis"
    )
    custom_glossary: Optional[str] = Field(
        default=None,
        description="Name of custom glossary to apply"
    )
    
    @classmethod
    def from_preset(cls, preset: str) -> "ProcessingSettings":
        """Create settings from a preset"""
        presets = {
            "default": {},
            "aggressive": {
                "merge_mode": MergeMode.AGGRESSIVE,
                "max_cps": 20.0,
                "min_duration_ms": 1000,
            },
            "conservative": {
                "merge_mode": MergeMode.CONSERVATIVE,
                "max_cps": 15.0,
                "min_duration_ms": 2000,
            },
            "accessibility": {
                "max_cps": 12.0,
                "min_duration_ms": 2000,
                "max_duration_ms": 8000,
                "line_wrap": 37,
            },
            "streaming": {
                "merge_mode": MergeMode.SMART,
                "max_cps": 18.0,
                "line_wrap": 42,
                "max_lines": 2,
            },
            "ml_enhanced": {
                "merge_mode": MergeMode.SMART,
                "max_cps": 17.0,
                "ml_enabled": True,
                "ml_models": {
                    "punctuation": True,
                    "grammar": True,
                    "contextual_fillers": True,
                    "scoring": True
                },
            },
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        return cls(**presets[preset])


class SettingsPreset(BaseModel):
    """Predefined settings preset"""
    name: str
    description: str
    settings: ProcessingSettings


class UpdateSettingsRequest(BaseModel):
    """Request to update processing settings"""
    preset: Optional[str] = Field(default=None, description="Use a preset")
    settings: Optional[ProcessingSettings] = Field(default=None, description="Custom settings")
    
    def get_settings(self) -> ProcessingSettings:
        """Get the final settings"""
        if self.preset:
            return ProcessingSettings.from_preset(self.preset)
        elif self.settings:
            return self.settings
        else:
            return ProcessingSettings()