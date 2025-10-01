from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import re

from app.domain.constants import Language


class GlossaryTerm(BaseModel):
    """Single glossary term"""
    original: str = Field(min_length=1, description="Original term to match")
    replacement: str = Field(min_length=1, description="Replacement term")
    case_sensitive: bool = Field(default=False)
    whole_word: bool = Field(default=True)
    context_regex: Optional[str] = Field(default=None, description="Optional regex for context matching")
    notes: Optional[str] = Field(default=None)
    
    @field_validator("context_regex")
    def validate_regex(cls, v):
        if v:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Invalid regex: {e}")
        return v


class CreateGlossaryRequest(BaseModel):
    """Request to create a new glossary"""
    name: str = Field(min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    language: Language
    terms: List[GlossaryTerm] = Field(min_length=1)
    tags: List[str] = Field(default_factory=list)
    
    @field_validator("terms")
    def validate_terms(cls, v):
        # Check for duplicate originals
        originals = [term.original.lower() for term in v]
        if len(originals) != len(set(originals)):
            raise ValueError("Duplicate original terms found")
        return v


class UpdateGlossaryRequest(BaseModel):
    """Request to update an existing glossary"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    terms: Optional[List[GlossaryTerm]] = Field(default=None, min_length=1)
    tags: Optional[List[str]] = Field(default=None)
    is_active: Optional[bool] = Field(default=None)


class GlossaryResponse(BaseModel):
    """Glossary response"""
    id: str
    name: str
    description: Optional[str]
    language: Language
    terms: List[GlossaryTerm]
    tags: List[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    term_count: int
    
    @classmethod
    def from_domain(cls, glossary: Dict[str, Any]) -> "GlossaryResponse":
        """Create from domain object"""
        return cls(
            id=glossary["id"],
            name=glossary["name"],
            description=glossary.get("description"),
            language=glossary["language"],
            terms=glossary["terms"],
            tags=glossary.get("tags", []),
            is_active=glossary.get("is_active", True),
            created_at=glossary["created_at"],
            updated_at=glossary["updated_at"],
            term_count=len(glossary["terms"]),
        )


class GlossaryListResponse(BaseModel):
    """List of glossaries"""
    glossaries: List[GlossaryResponse]
    total: int
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    

class GlossarySearchRequest(BaseModel):
    """Search for glossaries"""
    query: Optional[str] = Field(default=None, description="Search in name and description")
    language: Optional[Language] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)
    is_active: Optional[bool] = Field(default=None)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class ApplyGlossaryRequest(BaseModel):
    """Request to apply glossary to text"""
    text: str = Field(min_length=1)
    glossary_id: str
    preview: bool = Field(default=False, description="Preview changes without applying")


class ApplyGlossaryResponse(BaseModel):
    """Response from glossary application"""
    original_text: Optional[str] = None
    modified_text: str
    replacements_made: int
    changes: List[Dict[str, Any]] = Field(default_factory=list)