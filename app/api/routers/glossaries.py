from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List

from app.core.security import require_api_key
from app.core.rate_limit import standard_rate_limit
from app.api.schemas.glossaries import (
    CreateGlossaryRequest,
    UpdateGlossaryRequest,
    GlossaryResponse,
    GlossaryListResponse,
    ApplyGlossaryRequest,
    ApplyGlossaryResponse,
)
from app.domain.constants import Language
from app.services.glossary import glossary_store, glossary_enforcer

router = APIRouter()


@router.get(
    "/",
    response_model=GlossaryListResponse,
    dependencies=[Depends(require_api_key), Depends(standard_rate_limit)]
)
async def list_glossaries(
    language: Optional[Language] = Query(None),
    tags: Optional[List[str]] = Query(None),
    is_active: Optional[bool] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
) -> GlossaryListResponse:
    """List all glossaries with optional filtering"""
    glossaries, total = glossary_store.list_glossaries(
        language=language,
        tags=tags,
        is_active=is_active,
        page=page,
        page_size=page_size
    )
    
    return GlossaryListResponse(
        glossaries=[
            GlossaryResponse.from_domain(g) for g in glossaries
        ],
        total=total,
        page=page,
        page_size=page_size
    )


@router.post(
    "/",
    response_model=GlossaryResponse,
    dependencies=[Depends(require_api_key), Depends(standard_rate_limit)]
)
async def create_glossary(request: CreateGlossaryRequest) -> GlossaryResponse:
    """Create a new glossary"""
    glossary_id = glossary_store.create_glossary(
        name=request.name,
        language=request.language,
        terms=request.terms,
        description=request.description,
        tags=request.tags
    )
    
    glossary = glossary_store.get_glossary(glossary_id)
    if not glossary:
        raise HTTPException(status_code=500, detail="Failed to create glossary")
    
    return GlossaryResponse.from_domain(glossary)


@router.get(
    "/{glossary_id}",
    response_model=GlossaryResponse,
    dependencies=[Depends(require_api_key), Depends(standard_rate_limit)]
)
async def get_glossary(glossary_id: str) -> GlossaryResponse:
    """Get a specific glossary by ID"""
    glossary = glossary_store.get_glossary(glossary_id)
    if not glossary:
        raise HTTPException(status_code=404, detail="Glossary not found")
    
    return GlossaryResponse.from_domain(glossary)


@router.patch(
    "/{glossary_id}",
    response_model=GlossaryResponse,
    dependencies=[Depends(require_api_key), Depends(standard_rate_limit)]
)
async def update_glossary(
    glossary_id: str,
    request: UpdateGlossaryRequest
) -> GlossaryResponse:
    """Update an existing glossary"""
    success = glossary_store.update_glossary(
        glossary_id,
        name=request.name,
        description=request.description,
        terms=request.terms,
        tags=request.tags,
        is_active=request.is_active
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Glossary not found")
    
    glossary = glossary_store.get_glossary(glossary_id)
    return GlossaryResponse.from_domain(glossary)


@router.delete(
    "/{glossary_id}",
    dependencies=[Depends(require_api_key), Depends(standard_rate_limit)]
)
async def delete_glossary(glossary_id: str):
    """Delete a glossary"""
    success = glossary_store.delete_glossary(glossary_id)
    if not success:
        raise HTTPException(status_code=404, detail="Glossary not found")
    
    return {"message": "Glossary deleted successfully"}


@router.post(
    "/apply",
    response_model=ApplyGlossaryResponse,
    dependencies=[Depends(require_api_key), Depends(standard_rate_limit)]
)
async def apply_glossary(request: ApplyGlossaryRequest) -> ApplyGlossaryResponse:
    """Apply a glossary to text"""
    modified_text, changes = glossary_enforcer.apply_glossary(
        request.text,
        request.glossary_id,
        preview=request.preview
    )
    
    return ApplyGlossaryResponse(
        original_text=request.text if request.preview else None,
        modified_text=modified_text,
        replacements_made=len(changes),
        changes=changes
    )