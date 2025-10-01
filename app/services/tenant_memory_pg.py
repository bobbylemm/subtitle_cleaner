"""PostgreSQL-backed Per-Tenant Memory - Layer 6 Implementation"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from sqlalchemy import select, update, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.tenant_memory import ConfidenceLevel, CorrectionSource
from app.infra.models import (
    TenantCorrection, 
    TenantContext, 
    CorrectionHistory,
    CorrectionSourceEnum,
    ConfidenceLevelEnum
)

logger = logging.getLogger(__name__)


class TenantMemoryPostgreSQL:
    """
    PostgreSQL-backed tenant memory service.
    Replaces file-based storage with database persistence.
    """
    
    def __init__(self, max_lexicon_size: int = 10000):
        self.max_lexicon_size = max_lexicon_size
        
        # Confidence progression thresholds
        self.confidence_thresholds = {
            ConfidenceLevel.NEW: 0,
            ConfidenceLevel.LOW: 1,
            ConfidenceLevel.MEDIUM: 3,
            ConfidenceLevel.HIGH: 6,
            ConfidenceLevel.ESTABLISHED: 10
        }
    
    def _map_source_to_enum(self, source: CorrectionSource) -> str:
        """Map CorrectionSource to CorrectionSourceEnum"""
        mapping = {
            CorrectionSource.USER_FEEDBACK: CorrectionSourceEnum.USER_FEEDBACK.value,
            CorrectionSource.CONTEXT_MATCH: CorrectionSourceEnum.CONTEXT_MATCH.value,
            CorrectionSource.RETRIEVAL: CorrectionSourceEnum.RETRIEVAL.value,
            CorrectionSource.LLM_SELECTION: CorrectionSourceEnum.LLM_SELECTION.value,
            CorrectionSource.CONSENSUS: CorrectionSourceEnum.MANUAL.value
        }
        return mapping.get(source, CorrectionSourceEnum.MANUAL.value)
    
    def _map_confidence_to_enum(self, confidence: ConfidenceLevel) -> str:
        """Map ConfidenceLevel to ConfidenceLevelEnum"""
        mapping = {
            ConfidenceLevel.NEW: ConfidenceLevelEnum.LOW.value,
            ConfidenceLevel.LOW: ConfidenceLevelEnum.LOW.value,
            ConfidenceLevel.MEDIUM: ConfidenceLevelEnum.MEDIUM.value,
            ConfidenceLevel.HIGH: ConfidenceLevelEnum.HIGH.value,
            ConfidenceLevel.ESTABLISHED: ConfidenceLevelEnum.VERY_HIGH.value
        }
        return mapping.get(confidence, ConfidenceLevelEnum.MEDIUM.value)
    
    def _calculate_confidence_level(
        self,
        occurrence_count: int,
        boost: float = 0.0
    ) -> ConfidenceLevel:
        """Calculate confidence level based on occurrences"""
        effective_count = occurrence_count * (1 + boost)
        
        if effective_count >= self.confidence_thresholds[ConfidenceLevel.ESTABLISHED]:
            return ConfidenceLevel.ESTABLISHED
        elif effective_count >= self.confidence_thresholds[ConfidenceLevel.HIGH]:
            return ConfidenceLevel.HIGH
        elif effective_count >= self.confidence_thresholds[ConfidenceLevel.MEDIUM]:
            return ConfidenceLevel.MEDIUM
        elif effective_count >= self.confidence_thresholds[ConfidenceLevel.LOW]:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.NEW
    
    def _confidence_to_float(self, confidence: ConfidenceLevel) -> float:
        """Convert confidence level to float score"""
        mapping = {
            ConfidenceLevel.NEW: 0.3,
            ConfidenceLevel.LOW: 0.5,
            ConfidenceLevel.MEDIUM: 0.7,
            ConfidenceLevel.HIGH: 0.85,
            ConfidenceLevel.ESTABLISHED: 0.95
        }
        return mapping.get(confidence, 0.5)
    
    async def learn_correction(
        self,
        session: AsyncSession,
        tenant_id: str,
        original: str,
        corrected: str,
        source: CorrectionSource,
        context: str = "",
        confidence_boost: float = 0.0
    ):
        """Learn a new correction or reinforce existing one"""
        
        # Check if correction already exists
        stmt = select(TenantCorrection).where(
            and_(
                TenantCorrection.tenant_id == tenant_id,
                TenantCorrection.original_text == original
            )
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update existing correction
            existing.usage_count += 1
            existing.last_used_at = datetime.utcnow()
            
            # Update confidence
            confidence_level = self._calculate_confidence_level(
                existing.usage_count, 
                confidence_boost
            )
            existing.confidence = self._confidence_to_float(confidence_level)
            existing.confidence_level = self._map_confidence_to_enum(confidence_level)
            
            # Update context if different
            if context and context != existing.context:
                existing.context = context
            
            # If corrected text changed, update it
            if existing.corrected_text != corrected:
                logger.info(
                    f"Updating correction for '{original}': "
                    f"'{existing.corrected_text}' -> '{corrected}'"
                )
                existing.corrected_text = corrected
            
            existing.updated_at = datetime.utcnow()
        else:
            # Create new correction
            confidence_level = ConfidenceLevel.NEW
            new_correction = TenantCorrection(
                tenant_id=tenant_id,
                original_text=original,
                corrected_text=corrected,
                source=self._map_source_to_enum(source),
                confidence=self._confidence_to_float(confidence_level),
                confidence_level=self._map_confidence_to_enum(confidence_level),
                usage_count=1,
                success_count=0,
                context=context if context else None,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            session.add(new_correction)
        
        await session.commit()
        
        logger.debug(
            f"Learned correction for tenant {tenant_id}: '{original}' -> '{corrected}'"
        )
    
    async def apply_tenant_corrections(
        self,
        session: AsyncSession,
        tenant_id: str,
        segments: List[Dict],
        min_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """Apply tenant-specific corrections to segments"""
        
        min_confidence_float = self._confidence_to_float(min_confidence)
        
        # Get high-confidence corrections for this tenant
        stmt = select(TenantCorrection).where(
            and_(
                TenantCorrection.tenant_id == tenant_id,
                TenantCorrection.confidence >= min_confidence_float
            )
        ).order_by(TenantCorrection.confidence.desc())
        
        result = await session.execute(stmt)
        corrections_db = result.scalars().all()
        
        if not corrections_db:
            return segments, {}
        
        # Build correction map
        correction_map = {}
        for correction in corrections_db:
            correction_map[correction.original_text] = correction.corrected_text
            # Add lowercase variant
            correction_map[correction.original_text.lower()] = correction.corrected_text
        
        # Apply corrections to segments
        corrected_segments = []
        corrections_applied = {}
        
        for segment in segments:
            text = segment.get('text', '')
            corrected_text = text
            
            # Apply corrections
            for original, corrected in correction_map.items():
                if original in corrected_text:
                    corrected_text = corrected_text.replace(original, corrected)
                    corrections_applied[original] = corrected
                    
                    # Record usage
                    await self._record_correction_usage(
                        session, tenant_id, original, corrected
                    )
            
            # Create corrected segment
            corrected_segment = segment.copy()
            corrected_segment['text'] = corrected_text
            
            if corrected_text != text:
                corrected_segment['tenant_corrected'] = True
                corrected_segment['corrections'] = [
                    {"from": orig, "to": corr}
                    for orig, corr in corrections_applied.items()
                    if orig in text
                ]
            
            corrected_segments.append(corrected_segment)
        
        await session.commit()
        
        return corrected_segments, corrections_applied
    
    async def _record_correction_usage(
        self,
        session: AsyncSession,
        tenant_id: str,
        original: str,
        corrected: str
    ):
        """Record that a correction was used"""
        
        # Update usage_count and last_used_at
        stmt = (
            update(TenantCorrection)
            .where(
                and_(
                    TenantCorrection.tenant_id == tenant_id,
                    TenantCorrection.original_text == original
                )
            )
            .values(
                usage_count=TenantCorrection.usage_count + 1,
                last_used_at=datetime.utcnow()
            )
        )
        await session.execute(stmt)
    
    async def get_tenant_suggestions(
        self,
        session: AsyncSession,
        tenant_id: str,
        text: str,
        max_suggestions: int = 5
    ) -> List[Dict[str, Any]]:
        """Get correction suggestions based on tenant history"""
        
        # Find similar corrections using ILIKE
        stmt = select(TenantCorrection).where(
            and_(
                TenantCorrection.tenant_id == tenant_id,
                or_(
                    TenantCorrection.original_text.ilike(f"%{text}%"),
                    TenantCorrection.corrected_text.ilike(f"%{text}%")
                )
            )
        ).order_by(
            TenantCorrection.confidence.desc(),
            TenantCorrection.usage_count.desc()
        ).limit(max_suggestions)
        
        result = await session.execute(stmt)
        corrections = result.scalars().all()
        
        suggestions = []
        for correction in corrections:
            suggestions.append({
                "original": text,
                "suggestion": correction.corrected_text,
                "confidence": correction.confidence_level,
                "occurrences": correction.usage_count
            })
        
        return suggestions
    
    async def export_tenant_data(
        self,
        session: AsyncSession,
        tenant_id: str,
        include_low_confidence: bool = False
    ) -> Dict:
        """Export tenant data for backup or analysis"""
        
        # Build query
        stmt = select(TenantCorrection).where(
            TenantCorrection.tenant_id == tenant_id
        )
        
        if not include_low_confidence:
            min_confidence = self._confidence_to_float(ConfidenceLevel.MEDIUM)
            stmt = stmt.where(TenantCorrection.confidence >= min_confidence)
        
        stmt = stmt.order_by(TenantCorrection.confidence.desc())
        
        result = await session.execute(stmt)
        corrections = result.scalars().all()
        
        return {
            "tenant_id": tenant_id,
            "export_date": datetime.utcnow().isoformat(),
            "corrections_count": len(corrections),
            "corrections": [
                {
                    "original": c.original_text,
                    "corrected": c.corrected_text,
                    "confidence": c.confidence,
                    "confidence_level": c.confidence_level,
                    "usage_count": c.usage_count,
                    "source": c.source,
                    "context": c.context,
                    "created_at": c.created_at.isoformat(),
                    "last_used_at": c.last_used_at.isoformat() if c.last_used_at else None
                }
                for c in corrections
            ]
        }
    
    async def get_tenant_stats(
        self,
        session: AsyncSession,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Get statistics about tenant corrections"""
        
        # Count total corrections
        count_stmt = select(func.count(TenantCorrection.id)).where(
            TenantCorrection.tenant_id == tenant_id
        )
        count_result = await session.execute(count_stmt)
        total_corrections = count_result.scalar()
        
        # Count by confidence level
        confidence_stmt = select(
            TenantCorrection.confidence_level,
            func.count(TenantCorrection.id)
        ).where(
            TenantCorrection.tenant_id == tenant_id
        ).group_by(TenantCorrection.confidence_level)
        
        confidence_result = await session.execute(confidence_stmt)
        confidence_breakdown = dict(confidence_result.fetchall())
        
        # Total usage count
        usage_stmt = select(
            func.sum(TenantCorrection.usage_count)
        ).where(
            TenantCorrection.tenant_id == tenant_id
        )
        usage_result = await session.execute(usage_stmt)
        total_usage = usage_result.scalar() or 0
        
        return {
            "tenant_id": tenant_id,
            "total_corrections": total_corrections,
            "total_usage": total_usage,
            "confidence_breakdown": confidence_breakdown
        }