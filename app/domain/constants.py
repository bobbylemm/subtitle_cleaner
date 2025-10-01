from enum import Enum
from typing import Dict, List, Set


class SubtitleFormat(str, Enum):
    """Supported subtitle formats"""
    SRT = "srt"
    WEBVTT = "webvtt"


class Language(str, Enum):
    """Supported languages"""
    EN = "en"  # English
    ES = "es"  # Spanish
    FR = "fr"  # French
    DE = "de"  # German
    IT = "it"  # Italian
    PT = "pt"  # Portuguese
    NL = "nl"  # Dutch


class ProcessingStep(str, Enum):
    """Processing pipeline steps"""
    PARSE = "parse"
    VALIDATE = "validate"
    MERGE = "merge"
    WRAP = "wrap"
    NORMALIZE = "normalize"
    REMOVE_FILLERS = "remove_fillers"
    ENFORCE_GLOSSARY = "enforce_glossary"
    GUARDRAILS = "guardrails"
    SERIALIZE = "serialize"


class MergeMode(str, Enum):
    """Segment merge strategies"""
    OFF = "off"
    CONSERVATIVE = "conservative"
    SMART = "smart"
    AGGRESSIVE = "aggressive"


class FillerMode(str, Enum):
    """Filler handling strategies"""
    KEEP = "keep"
    REMOVE = "remove"
    SMART = "smart"


# Language-specific filler words/phrases
FILLERS: Dict[Language, Set[str]] = {
    Language.EN: {
        "um", "uh", "er", "ah", "like", "you know", "I mean", "sort of", 
        "kind of", "basically", "actually", "literally", "right", "okay",
        "so", "well", "hmm", "uhh", "umm", "eh", "oh", "yeah"
    },
    Language.ES: {
        "eh", "este", "pues", "bueno", "o sea", "es decir", "mira",
        "sabes", "vale", "verdad", "vaya", "hombre", "entonces"
    },
    Language.FR: {
        "euh", "ben", "bah", "alors", "donc", "voila", "quoi", "hein",
        "enfin", "bon", "eh bien", "tu vois", "tu sais", "genre"
    },
    Language.DE: {
        "aeh", "aehm", "also", "halt", "eben", "ja", "ne", "sozusagen",
        "quasi", "eigentlich", "irgendwie", "praktisch", "nun"
    },
    Language.IT: {
        "eh", "ehm", "cioe", "beh", "allora", "quindi", "ecco", "vabbe",
        "diciamo", "praticamente", "sostanzialmente", "insomma"
    },
    Language.PT: {
        "eh", "ah", "hum", "ne", "pois", "entao", "tipo", "assim",
        "quer dizer", "olha", "veja", "sabe", "entende"
    },
    Language.NL: {
        "eh", "uhm", "nou", "dus", "eigenlijk", "zeg maar", "weet je",
        "he", "toch", "gewoon", "even", "ff", "ofzo"
    },
}

# Punctuation normalization rules
PUNCTUATION_RULES = {
    "...": "…",  # Use ellipsis character
    "!!": "!",   # Single exclamation
    "??": "?",   # Single question mark
    "  ": " ",   # Remove double spaces
}

# Maximum file sizes
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_PREVIEW_SEGMENTS = 50
MAX_BATCH_SIZE = 100

# Timing constraints (milliseconds)
MIN_SEGMENT_DURATION_MS = 500   # 0.5s absolute minimum
MAX_SEGMENT_DURATION_MS = 10000  # 10s absolute maximum
DEFAULT_MIN_DURATION_MS = 1500   # 1.5s default
DEFAULT_MAX_DURATION_MS = 6000   # 6s default

# Text constraints
DEFAULT_CPS = 17.0
MAX_CPS = 25.0
MIN_CPS = 5.0
DEFAULT_LINE_LENGTH = 42
MAX_LINE_LENGTH = 80
MIN_LINE_LENGTH = 20
MAX_LINES_PER_SEGMENT = 2

# Cache TTLs (seconds)
CACHE_TTL_GLOSSARY = 3600  # 1 hour
CACHE_TTL_PREVIEW = 300    # 5 minutes
CACHE_TTL_VALIDATION = 60  # 1 minute

# Processing timeouts (seconds)
PROCESSING_TIMEOUT = 60
PREVIEW_TIMEOUT = 10
VALIDATION_TIMEOUT = 30

# Regex patterns
TIMECODE_PATTERN_SRT = r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
TIMECODE_PATTERN_WEBVTT = r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
TAG_PATTERN = r"<[^>]+>"
MULTIPLE_SPACES_PATTERN = r"\s+"