"""Add tenant memory tables

Revision ID: 001
Revises: 
Create Date: 2025-09-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create tenant_corrections table
    op.create_table(
        'tenant_corrections',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('tenant_id', sa.String(length=255), nullable=False),
        sa.Column('original_text', sa.String(length=500), nullable=False),
        sa.Column('corrected_text', sa.String(length=500), nullable=False),
        sa.Column('source', sa.String(length=50), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('confidence_level', sa.String(length=20), nullable=False, server_default='medium'),
        sa.Column('usage_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('success_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('context', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for tenant_corrections
    op.create_index('ix_tenant_corrections_tenant_id', 'tenant_corrections', ['tenant_id'])
    op.create_index('ix_tenant_original', 'tenant_corrections', ['tenant_id', 'original_text'])
    op.create_index('ix_tenant_confidence', 'tenant_corrections', ['tenant_id', 'confidence'])
    op.create_index('ix_tenant_source', 'tenant_corrections', ['tenant_id', 'source'])
    
    # Create tenant_context table
    op.create_table(
        'tenant_context',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('tenant_id', sa.String(length=255), nullable=False),
        sa.Column('entity_text', sa.String(length=500), nullable=False),
        sa.Column('entity_type', sa.String(length=100), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('authority_score', sa.Float(), nullable=False, server_default='0.8'),
        sa.Column('source_id', sa.String(length=255), nullable=True),
        sa.Column('source_type', sa.String(length=100), nullable=True),
        sa.Column('usage_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for tenant_context
    op.create_index('ix_tenant_context_tenant_id', 'tenant_context', ['tenant_id'])
    op.create_index('ix_tenant_entity', 'tenant_context', ['tenant_id', 'entity_text'])
    op.create_index('ix_tenant_entity_type', 'tenant_context', ['tenant_id', 'entity_type'])
    
    # Create correction_history table
    op.create_table(
        'correction_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('tenant_id', sa.String(length=255), nullable=False),
        sa.Column('correction_id', sa.Integer(), nullable=True),
        sa.Column('original_text', sa.String(length=500), nullable=False),
        sa.Column('corrected_text', sa.String(length=500), nullable=False),
        sa.Column('was_successful', sa.Boolean(), nullable=True),
        sa.Column('user_feedback', sa.Text(), nullable=True),
        sa.Column('segment_context', sa.Text(), nullable=True),
        sa.Column('language', sa.String(length=10), nullable=True),
        sa.Column('applied_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for correction_history
    op.create_index('ix_correction_history_tenant_id', 'correction_history', ['tenant_id'])
    op.create_index('ix_correction_history_applied_at', 'correction_history', ['applied_at'])
    op.create_index('ix_tenant_applied', 'correction_history', ['tenant_id', 'applied_at'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index('ix_tenant_applied', table_name='correction_history')
    op.drop_index('ix_correction_history_applied_at', table_name='correction_history')
    op.drop_index('ix_correction_history_tenant_id', table_name='correction_history')
    op.drop_table('correction_history')
    
    op.drop_index('ix_tenant_entity_type', table_name='tenant_context')
    op.drop_index('ix_tenant_entity', table_name='tenant_context')
    op.drop_index('ix_tenant_context_tenant_id', table_name='tenant_context')
    op.drop_table('tenant_context')
    
    op.drop_index('ix_tenant_source', table_name='tenant_corrections')
    op.drop_index('ix_tenant_confidence', table_name='tenant_corrections')
    op.drop_index('ix_tenant_original', table_name='tenant_corrections')
    op.drop_index('ix_tenant_corrections_tenant_id', table_name='tenant_corrections')
    op.drop_table('tenant_corrections')