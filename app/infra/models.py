"""Database models for tenant memory persistence"""

from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, Integer, DateTime, Text, Index, Enum as SQLEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import enum


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


class CorrectionSourceEnum(str, enum.Enum):
    """Source of correction"""
    CONTEXT_MATCH = "context_match"
    RETRIEVAL = "retrieval"
    LLM_SELECTION = "llm_selection"
    USER_FEEDBACK = "user_feedback"
    MANUAL = "manual"


class ConfidenceLevelEnum(str, enum.Enum):
    """Confidence level for corrections"""
    LOW = "low"           # 0.0-0.5
    MEDIUM = "medium"     # 0.5-0.75
    HIGH = "high"         # 0.75-0.9
    VERY_HIGH = "very_high"  # 0.9-1.0


class TenantCorrection(Base):
    """
    Stores learned corrections per tenant.
    Used by Layer 6 (Tenant Memory) for persistent learning.
    """
    __tablename__ = "tenant_corrections"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Tenant identification
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    # Correction data
    original_text: Mapped[str] = mapped_column(String(500), nullable=False)
    corrected_text: Mapped[str] = mapped_column(String(500), nullable=False)
    
    # Metadata
    source: Mapped[str] = mapped_column(
        SQLEnum(CorrectionSourceEnum, native_enum=False),
        nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    confidence_level: Mapped[str] = mapped_column(
        SQLEnum(ConfidenceLevelEnum, native_enum=False),
        nullable=False,
        default=ConfidenceLevelEnum.MEDIUM
    )
    
    # Usage tracking
    usage_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Context
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        nullable=False, 
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Composite indexes for performance
    __table_args__ = (
        Index('ix_tenant_original', 'tenant_id', 'original_text'),
        Index('ix_tenant_confidence', 'tenant_id', 'confidence'),
        Index('ix_tenant_source', 'tenant_id', 'source'),
    )
    
    def __repr__(self) -> str:
        return (
            f"<TenantCorrection(tenant={self.tenant_id}, "
            f"'{self.original_text}' -> '{self.corrected_text}', "
            f"confidence={self.confidence:.2f})>"
        )


class TenantContext(Base):
    """
    Stores tenant-specific context and glossary terms.
    Used for persistent context across sessions.
    """
    __tablename__ = "tenant_context"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Tenant identification
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    # Context data
    entity_text: Mapped[str] = mapped_column(String(500), nullable=False)
    entity_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Metadata
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    authority_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.8)
    
    # Source tracking
    source_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Usage tracking
    usage_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    __table_args__ = (
        Index('ix_tenant_entity', 'tenant_id', 'entity_text'),
        Index('ix_tenant_entity_type', 'tenant_id', 'entity_type'),
    )
    
    def __repr__(self) -> str:
        return (
            f"<TenantContext(tenant={self.tenant_id}, "
            f"entity='{self.entity_text}', type={self.entity_type})>"
        )


class CorrectionHistory(Base):
    """
    Tracks correction performance over time.
    Used for analytics and continuous improvement.
    """
    __tablename__ = "correction_history"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Tenant identification
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    # Correction reference
    correction_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Reference to tenant_corrections.id"
    )
    
    # Applied correction
    original_text: Mapped[str] = mapped_column(String(500), nullable=False)
    corrected_text: Mapped[str] = mapped_column(String(500), nullable=False)
    
    # Outcome
    was_successful: Mapped[Optional[bool]] = mapped_column(nullable=True)
    user_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Context
    segment_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    
    # Timestamp
    applied_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True
    )
    
    __table_args__ = (
        Index('ix_tenant_applied', 'tenant_id', 'applied_at'),
    )
    
    def __repr__(self) -> str:
        return (
            f"<CorrectionHistory(tenant={self.tenant_id}, "
            f"'{self.original_text}' -> '{self.corrected_text}', "
            f"success={self.was_successful})>"
        )