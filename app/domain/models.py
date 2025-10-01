from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class Segment:
    """Core subtitle segment"""
    idx: int
    start_ms: int
    end_ms: int
    text: str
    
    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms
    
    @property
    def cps(self) -> float:
        """Characters per second"""
        if self.duration_ms == 0:
            return 0
        return len(self.text) / (self.duration_ms / 1000)
    
    def overlaps_with(self, other: "Segment") -> bool:
        return not (self.end_ms <= other.start_ms or self.start_ms >= other.end_ms)


@dataclass
class SubtitleDocument:
    """Container for subtitle segments"""
    segments: List[Segment]
    
    def find_overlaps(self) -> List[tuple[int, int]]:
        """Find overlapping segment pairs"""
        overlaps = []
        for i in range(len(self.segments)):
            for j in range(i + 1, len(self.segments)):
                if self.segments[i].overlaps_with(self.segments[j]):
                    overlaps.append((i, j))
        return overlaps
    
    @property
    def duration_ms(self) -> int:
        if not self.segments:
            return 0
        return self.segments[-1].end_ms


@dataclass
class Settings:
    """Processing settings with sane defaults"""
    merge_mode: str = "smart"  # smart, aggressive, conservative, off
    line_wrap: int = 42  # characters per line
    max_lines: int = 2
    min_duration_ms: int = 1500  # 1.5 seconds
    max_duration_ms: int = 6000  # 6 seconds
    max_cps: float = 17.0  # characters per second
    filler_mode: str = "remove"  # remove, keep, smart
    language: str = "en"
    glossary: Optional[str] = None
    preserve_formatting: bool = False
    custom_fillers: List[str] = None
    
    # ML enhancement settings
    ml_enabled: bool = False
    ml_models: Optional[Dict[str, bool]] = None
    ml_constraints: Optional[Dict[str, float]] = None
    ml_device: str = "cpu"
    ml_quantized: bool = True
    
    # Enhanced cleaning features (Phase 1)
    enable_punctuation: bool = False
    enable_entity_stabilization: bool = False
    enable_context_extraction: bool = False
    
    # Phase 2: Contextual Understanding features
    enable_topic_segmentation: bool = False
    enable_speaker_tracking: bool = False
    enable_coreference_resolution: bool = False
    
    # Phase 3: Advanced Processing features
    enable_domain_classification: bool = False
    enable_quality_scoring: bool = False
    enable_cps_optimization: bool = False
    enable_vocabulary_enforcement: bool = False
    enable_adaptive_processing: bool = False
    custom_glossary: Optional[str] = None
    
    def __post_init__(self):
        if self.custom_fillers is None:
            self.custom_fillers = []
        if self.ml_models is None:
            self.ml_models = {
                "punctuation": True,
                "grammar": True,
                "contextual_fillers": True,
                "scoring": True
            }
        if self.ml_constraints is None:
            self.ml_constraints = {
                "max_edit_ratio": 0.15,
                "max_char_change": 8,
                "min_confidence": 0.7
            }
    
    @classmethod
    def from_preset(cls, preset: str) -> "Settings":
        """Create settings from a preset"""
        presets = {
            "default": {},
            "aggressive": {
                "merge_mode": "aggressive",
                "max_cps": 20.0,
                "min_duration_ms": 1000,
            },
            "conservative": {
                "merge_mode": "conservative",
                "max_cps": 15.0,
                "min_duration_ms": 2000,
            },
            "accessibility": {
                "max_cps": 12.0,
                "min_duration_ms": 2000,
                "max_duration_ms": 8000,
                "line_wrap": 37,
            },
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}")
        
        return cls(**presets[preset])