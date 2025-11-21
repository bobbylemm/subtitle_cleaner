import os
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Allow extra fields from environment
    )

    # Application settings
    APP_NAME: str = "subtitle-cleaner"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = Field("development", description="Application environment")
    DEBUG: bool = Field(False, description="Debug mode")
    PIPELINE_VERSION: str = Field("1.0.0", env="PIPELINE_VERSION")

    # API settings
    API_PREFIX: str = "/v1"
    CORS_ORIGINS: str = "*"
    PORT: int = 8080

    # Security
    API_KEYS: str = Field(..., description="Comma-separated list of valid API keys")
    API_KEY_HEADER: str = "X-API-Key"
    _api_keys_list: List[str] = []

    @field_validator("API_KEYS")
    @classmethod
    def validate_api_keys(cls, v: str) -> str:
        if not v:
            raise ValueError("API_KEYS must be provided")
        keys = [key.strip() for key in v.split(",") if key.strip()]
        if not keys:
            raise ValueError("At least one API key must be provided")
        return v  # Return the original string, not the list

    # Database
    DATABASE_URL: str = Field(
        "postgresql://subtitle_user:subtitle_pass@localhost:5432/subtitle_db"
    )

    # Redis
    REDIS_URL: str = Field("redis://localhost:6379/0")
    REDIS_PASSWORD: Optional[str] = None
    REDIS_MAX_CONNECTIONS: int = 50

    # Universal Corrector Settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_MAX_TOKENS: int = 4096
    OPENAI_TEMPERATURE: float = 0.0

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = Field(100, description="Requests per period")
    RATE_LIMIT_PERIOD: int = Field(60, description="Period in seconds")
    RATE_LIMIT_CONCURRENT: int = Field(10, description="Max concurrent requests per key")
    RATE_LIMIT_BURST: int = Field(20, description="Burst allowance")

    # File processing limits
    MAX_FILE_SIZE_MB: int = Field(10, description="Maximum file size in MB")
    MAX_SEGMENTS: int = Field(10000, description="Maximum number of segments")
    MAX_PROCESSING_TIME_S: int = Field(60, description="Maximum processing time in seconds")

    # Default processing settings
    DEFAULT_LANGUAGE: str = "en"
    SUPPORTED_LANGUAGES: str = "en,es,fr,de,it,pt,nl"
    DEFAULT_PRESET: str = "standard"

    @field_validator("SUPPORTED_LANGUAGES")
    @classmethod
    def parse_languages(cls, v: str) -> List[str]:
        return [lang.strip() for lang in v.split(",") if lang.strip()]

    # Subtitle processing defaults
    MERGE_THRESHOLD_MS: int = Field(100, description="Threshold for merging segments (ms)")
    MIN_SEGMENT_DURATION_MS: int = Field(1500, description="Minimum segment duration (ms)")
    MAX_SEGMENT_DURATION_MS: int = Field(6000, description="Maximum segment duration (ms)")
    TARGET_CPS: float = Field(17.0, description="Target characters per second")
    MAX_CPS: float = Field(20.0, description="Maximum characters per second")
    MAX_LINE_LENGTH: int = Field(42, description="Maximum characters per line")
    MAX_LINES_PER_SEGMENT: int = Field(2, description="Maximum lines per segment")

    # Wrap settings
    WRAP_PREFER_PUNCTUATION: bool = True
    WRAP_BALANCED_LINES: bool = True
    WRAP_PRESERVE_PHRASES: bool = True

    # Normalisation settings
    NORMALIZE_PUNCTUATION: bool = True
    NORMALIZE_CASING: bool = True
    REMOVE_FILLERS: bool = True
    ENFORCE_GLOSSARY: bool = True
    FIX_COMMON_ERRORS: bool = True

    # Validation settings
    STRICT_VALIDATION: bool = False
    ALLOW_OVERLAPS: bool = False
    REQUIRE_SEQUENTIAL: bool = True

    # Observability
    METRICS_ENABLED: bool = True
    OTLP_ENDPOINT: Optional[str] = None
    OTLP_ENABLED: bool = False
    LOG_LEVEL: str = "info"
    LOG_FORMAT: str = "json"

    # Worker configuration
    WORKERS: int = Field(2, description="Number of worker processes")
    WORKER_CLASS: str = "uvloop"

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def api_keys_list(self) -> List[str]:
        if not self._api_keys_list:
            self._api_keys_list = [key.strip() for key in self.API_KEYS.split(",") if key.strip()]
        return self._api_keys_list

    @property
    def supported_languages_list(self) -> List[str]:
        return [lang.strip() for lang in self.SUPPORTED_LANGUAGES.split(",") if lang.strip()]

    def is_language_supported(self, language: str) -> bool:
        return language.lower() in self.supported_languages_list


settings = Settings()