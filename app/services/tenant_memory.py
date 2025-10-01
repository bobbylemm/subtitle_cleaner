"""Per-Tenant Memory - Layer 6 Implementation"""

import json
import logging
import hashlib
import asyncio
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import aiofiles
import redis.asyncio as redis
from cachetools import TTLCache

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    NEW = "new"              # Just learned
    LOW = "low"              # 1-2 occurrences  
    MEDIUM = "medium"        # 3-5 occurrences
    HIGH = "high"            # 6-10 occurrences
    ESTABLISHED = "established"  # 10+ occurrences


class CorrectionSource(Enum):
    USER_FEEDBACK = "user_feedback"      # Direct user correction
    CONTEXT_MATCH = "context_match"      # Matched from provided context
    RETRIEVAL = "retrieval"              # From retrieval layer
    LLM_SELECTION = "llm_selection"      # From LLM layer
    CONSENSUS = "consensus"              # Multiple sources agree


@dataclass
class LexiconEntry:
    original: str                   # Original incorrect form
    corrected: str                   # Corrected form
    confidence: ConfidenceLevel
    occurrence_count: int
    first_seen: datetime
    last_seen: datetime
    sources: Set[CorrectionSource]
    contexts: List[str]             # Sample contexts where seen
    phonetic_key: Optional[str] = None
    entity_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "original": self.original,
            "corrected": self.corrected,
            "confidence": self.confidence.value,
            "occurrence_count": self.occurrence_count,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "sources": list(s.value for s in self.sources),
            "contexts": self.contexts[:5],  # Limit stored contexts
            "phonetic_key": self.phonetic_key,
            "entity_type": self.entity_type,
            "metadata": self.metadata
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'LexiconEntry':
        """Create from dictionary"""
        return LexiconEntry(
            original=data["original"],
            corrected=data["corrected"],
            confidence=ConfidenceLevel(data["confidence"]),
            occurrence_count=data["occurrence_count"],
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            sources=set(CorrectionSource(s) for s in data["sources"]),
            contexts=data.get("contexts", []),
            phonetic_key=data.get("phonetic_key"),
            entity_type=data.get("entity_type"),
            metadata=data.get("metadata", {})
        )


@dataclass
class TenantProfile:
    tenant_id: str
    created_at: datetime
    domain: Optional[str] = None        # e.g., "politics", "technology"
    region: Optional[str] = None        # e.g., "NG", "US"
    language: str = "en"
    lexicon_size: int = 0
    total_corrections: int = 0
    last_activity: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TenantMemory:
    """Layer 6: Per-Tenant Memory for learning from corrections"""
    
    def __init__(self,
                 redis_client: Optional[redis.Redis] = None,
                 storage_path: Optional[Path] = None,
                 max_lexicon_size: int = 10000,
                 confidence_thresholds: Optional[Dict] = None,
                 ttl_days: int = 90):
        
        self.redis_client = redis_client
        self.storage_path = Path(storage_path) if storage_path else Path("./tenant_data")
        self.max_lexicon_size = max_lexicon_size
        self.ttl_days = ttl_days
        
        # Create storage directory if using file storage
        if not redis_client:
            self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Confidence progression thresholds
        self.confidence_thresholds = confidence_thresholds or {
            ConfidenceLevel.NEW: 0,
            ConfidenceLevel.LOW: 1,
            ConfidenceLevel.MEDIUM: 3,
            ConfidenceLevel.HIGH: 6,
            ConfidenceLevel.ESTABLISHED: 10
        }
        
        # In-memory cache for active tenants
        self.cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes
        
        # Phonetic processor for fuzzy matching
        try:
            import jellyfish
            self.phonetic_available = True
        except ImportError:
            self.phonetic_available = False
            logger.warning("jellyfish not available - phonetic matching disabled")
    
    async def get_tenant_lexicon(
        self,
        tenant_id: str,
        create_if_missing: bool = True
    ) -> Dict[str, LexiconEntry]:
        """Load tenant's lexicon"""
        
        # Check cache first
        cache_key = f"lexicon:{tenant_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        lexicon = {}
        
        if self.redis_client:
            # Load from Redis
            try:
                data = await self.redis_client.get(f"tenant:lexicon:{tenant_id}")
                if data:
                    lexicon_data = json.loads(data)
                    lexicon = {
                        k: LexiconEntry.from_dict(v)
                        for k, v in lexicon_data.items()
                    }
            except Exception as e:
                logger.error(f"Failed to load lexicon from Redis: {e}")
        else:
            # Load from file
            lexicon_file = self.storage_path / f"{tenant_id}_lexicon.json"
            if lexicon_file.exists():
                try:
                    async with aiofiles.open(lexicon_file, 'r') as f:
                        data = await f.read()
                        lexicon_data = json.loads(data)
                        lexicon = {
                            k: LexiconEntry.from_dict(v)
                            for k, v in lexicon_data.items()
                        }
                except Exception as e:
                    logger.error(f"Failed to load lexicon from file: {e}")
        
        # Cache the lexicon
        self.cache[cache_key] = lexicon
        
        return lexicon
    
    async def save_tenant_lexicon(
        self,
        tenant_id: str,
        lexicon: Dict[str, LexiconEntry]
    ):
        """Save tenant's lexicon"""
        
        # Convert to serializable format
        lexicon_data = {
            k: v.to_dict() for k, v in lexicon.items()
        }
        
        if self.redis_client:
            # Save to Redis with TTL
            try:
                await self.redis_client.setex(
                    f"tenant:lexicon:{tenant_id}",
                    timedelta(days=self.ttl_days),
                    json.dumps(lexicon_data)
                )
            except Exception as e:
                logger.error(f"Failed to save lexicon to Redis: {e}")
        else:
            # Save to file
            lexicon_file = self.storage_path / f"{tenant_id}_lexicon.json"
            try:
                async with aiofiles.open(lexicon_file, 'w') as f:
                    await f.write(json.dumps(lexicon_data, indent=2))
            except Exception as e:
                logger.error(f"Failed to save lexicon to file: {e}")
        
        # Update cache
        cache_key = f"lexicon:{tenant_id}"
        self.cache[cache_key] = lexicon
    
    async def learn_correction(
        self,
        tenant_id: str,
        original: str,
        corrected: str,
        source: CorrectionSource,
        context: str = "",
        confidence_boost: float = 0.0
    ):
        """Learn a new correction or reinforce existing one"""
        
        # Load tenant lexicon
        lexicon = await self.get_tenant_lexicon(tenant_id)
        
        # Create key for lookup
        key = self._create_key(original)
        
        now = datetime.now()
        
        if key in lexicon:
            # Update existing entry
            entry = lexicon[key]
            entry.occurrence_count += 1
            entry.last_seen = now
            entry.sources.add(source)
            
            # Add context if new
            if context and context not in entry.contexts:
                entry.contexts.append(context)
                # Keep only recent contexts
                entry.contexts = entry.contexts[-10:]
            
            # Update confidence level
            entry.confidence = self._calculate_confidence_level(
                entry.occurrence_count,
                len(entry.sources),
                confidence_boost
            )
            
            # Check if corrected form changed (conflict resolution)
            if entry.corrected != corrected:
                # Resolve conflict based on source priority
                if self._should_update_correction(entry, source):
                    logger.info(f"Updating correction for '{original}': '{entry.corrected}' -> '{corrected}'")
                    entry.corrected = corrected
        else:
            # Create new entry
            entry = LexiconEntry(
                original=original,
                corrected=corrected,
                confidence=ConfidenceLevel.NEW,
                occurrence_count=1,
                first_seen=now,
                last_seen=now,
                sources={source},
                contexts=[context] if context else [],
                phonetic_key=self._get_phonetic_key(original)
            )
            
            lexicon[key] = entry
        
        # Prune if lexicon too large
        if len(lexicon) > self.max_lexicon_size:
            lexicon = await self._prune_lexicon(lexicon)
        
        # Save updated lexicon
        await self.save_tenant_lexicon(tenant_id, lexicon)
        
        logger.debug(f"Learned correction for tenant {tenant_id}: '{original}' -> '{corrected}' (confidence: {entry.confidence.value})")
    
    async def apply_tenant_corrections(
        self,
        tenant_id: str,
        segments: List[Dict],
        min_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """Apply tenant-specific corrections to segments"""
        
        # Load tenant lexicon
        lexicon = await self.get_tenant_lexicon(tenant_id, create_if_missing=False)
        
        if not lexicon:
            return segments, {}
        
        # Filter by minimum confidence
        applicable_corrections = {
            k: v for k, v in lexicon.items()
            if self._confidence_to_int(v.confidence) >= self._confidence_to_int(min_confidence)
        }
        
        if not applicable_corrections:
            return segments, {}
        
        # Build correction map
        correction_map = {}
        for entry in applicable_corrections.values():
            correction_map[entry.original] = entry.corrected
            
            # Add lowercase variant
            correction_map[entry.original.lower()] = entry.corrected
            
            # Add phonetic variants if available
            if self.phonetic_available and entry.phonetic_key:
                # Find other entries with same phonetic key
                for other_entry in lexicon.values():
                    if other_entry.phonetic_key == entry.phonetic_key:
                        correction_map[other_entry.original] = entry.corrected
        
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
        
        return corrected_segments, corrections_applied
    
    async def get_tenant_suggestions(
        self,
        tenant_id: str,
        text: str,
        max_suggestions: int = 5
    ) -> List[Dict[str, str]]:
        """Get correction suggestions based on tenant history"""
        
        lexicon = await self.get_tenant_lexicon(tenant_id, create_if_missing=False)
        
        if not lexicon:
            return []
        
        suggestions = []
        text_lower = text.lower()
        
        # Find potential corrections
        for entry in lexicon.values():
            # Check if this might be a correction
            similarity = self._calculate_similarity(text, entry.original)
            
            if similarity > 0.7:  # Threshold for suggestion
                suggestions.append({
                    "original": text,
                    "suggestion": entry.corrected,
                    "confidence": entry.confidence.value,
                    "similarity": similarity,
                    "occurrences": entry.occurrence_count
                })
        
        # Sort by confidence and similarity
        suggestions.sort(
            key=lambda x: (self._confidence_to_int(ConfidenceLevel(x["confidence"])), x["similarity"]),
            reverse=True
        )
        
        return suggestions[:max_suggestions]
    
    def _calculate_confidence_level(
        self,
        occurrence_count: int,
        source_diversity: int,
        boost: float = 0.0
    ) -> ConfidenceLevel:
        """Calculate confidence level based on occurrences and sources"""
        
        # Apply boost
        effective_count = occurrence_count * (1 + boost)
        
        # Bonus for multiple sources
        if source_diversity > 2:
            effective_count *= 1.5
        
        # Determine level
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
    
    def _should_update_correction(
        self,
        entry: LexiconEntry,
        new_source: CorrectionSource
    ) -> bool:
        """Determine if correction should be updated based on source priority"""
        
        # Source priority (higher = more authoritative)
        source_priority = {
            CorrectionSource.USER_FEEDBACK: 5,
            CorrectionSource.CONSENSUS: 4,
            CorrectionSource.CONTEXT_MATCH: 3,
            CorrectionSource.LLM_SELECTION: 2,
            CorrectionSource.RETRIEVAL: 1
        }
        
        # Get highest priority existing source
        max_existing_priority = max(
            source_priority.get(s, 0) for s in entry.sources
        )
        
        new_priority = source_priority.get(new_source, 0)
        
        # Update if new source has higher or equal priority
        return new_priority >= max_existing_priority
    
    async def _prune_lexicon(
        self,
        lexicon: Dict[str, LexiconEntry]
    ) -> Dict[str, LexiconEntry]:
        """Prune lexicon to stay within size limits"""
        
        # Sort entries by importance (confidence, recency, frequency)
        entries = list(lexicon.items())
        entries.sort(
            key=lambda x: (
                self._confidence_to_int(x[1].confidence),
                x[1].occurrence_count,
                x[1].last_seen.timestamp()
            ),
            reverse=True
        )
        
        # Keep top entries
        pruned_lexicon = dict(entries[:self.max_lexicon_size])
        
        logger.info(f"Pruned lexicon from {len(lexicon)} to {len(pruned_lexicon)} entries")
        
        return pruned_lexicon
    
    def _create_key(self, text: str) -> str:
        """Create normalized key for lexicon lookup"""
        return text.lower().strip()
    
    def _get_phonetic_key(self, text: str) -> Optional[str]:
        """Get phonetic key for fuzzy matching"""
        if not self.phonetic_available:
            return None
        
        try:
            import jellyfish
            return jellyfish.metaphone(text)
        except:
            return None
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not self.phonetic_available:
            # Simple character-based similarity
            return self._simple_similarity(text1, text2)
        
        try:
            import jellyfish
            
            # Combine multiple similarity metrics
            lev_distance = jellyfish.levenshtein_distance(text1.lower(), text2.lower())
            max_len = max(len(text1), len(text2))
            lev_similarity = 1 - (lev_distance / max_len) if max_len > 0 else 0
            
            # Phonetic similarity
            phonetic_similarity = 1.0 if jellyfish.metaphone(text1) == jellyfish.metaphone(text2) else 0.3
            
            # Weighted average
            return (lev_similarity * 0.7) + (phonetic_similarity * 0.3)
        except:
            return self._simple_similarity(text1, text2)
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple similarity calculation without external libraries"""
        text1 = text1.lower()
        text2 = text2.lower()
        
        if text1 == text2:
            return 1.0
        
        # Character overlap
        common_chars = set(text1) & set(text2)
        if not common_chars:
            return 0.0
        
        return len(common_chars) / max(len(set(text1)), len(set(text2)))
    
    def _confidence_to_int(self, confidence: ConfidenceLevel) -> int:
        """Convert confidence level to integer for comparison"""
        level_map = {
            ConfidenceLevel.NEW: 0,
            ConfidenceLevel.LOW: 1,
            ConfidenceLevel.MEDIUM: 2,
            ConfidenceLevel.HIGH: 3,
            ConfidenceLevel.ESTABLISHED: 4
        }
        return level_map.get(confidence, 0)
    
    async def export_tenant_data(
        self,
        tenant_id: str,
        include_low_confidence: bool = False
    ) -> Dict:
        """Export tenant data for backup or analysis"""
        
        lexicon = await self.get_tenant_lexicon(tenant_id, create_if_missing=False)
        
        if not lexicon:
            return {}
        
        # Filter by confidence if requested
        if not include_low_confidence:
            lexicon = {
                k: v for k, v in lexicon.items()
                if self._confidence_to_int(v.confidence) >= self._confidence_to_int(ConfidenceLevel.MEDIUM)
            }
        
        return {
            "tenant_id": tenant_id,
            "export_date": datetime.now().isoformat(),
            "lexicon_size": len(lexicon),
            "entries": {k: v.to_dict() for k, v in lexicon.items()}
        }
    
    async def import_tenant_data(
        self,
        tenant_id: str,
        data: Dict,
        merge: bool = True
    ):
        """Import tenant data from backup"""
        
        if "entries" not in data:
            raise ValueError("Invalid import data - missing entries")
        
        # Parse entries
        imported_lexicon = {
            k: LexiconEntry.from_dict(v)
            for k, v in data["entries"].items()
        }
        
        if merge:
            # Merge with existing lexicon
            existing_lexicon = await self.get_tenant_lexicon(tenant_id)
            
            for key, entry in imported_lexicon.items():
                if key in existing_lexicon:
                    # Merge entries - keep higher confidence
                    existing = existing_lexicon[key]
                    if self._confidence_to_int(entry.confidence) > self._confidence_to_int(existing.confidence):
                        existing_lexicon[key] = entry
                    else:
                        # Update occurrence count
                        existing.occurrence_count += entry.occurrence_count
                        existing.sources.update(entry.sources)
                        existing.confidence = self._calculate_confidence_level(
                            existing.occurrence_count,
                            len(existing.sources)
                        )
                else:
                    existing_lexicon[key] = entry
            
            await self.save_tenant_lexicon(tenant_id, existing_lexicon)
        else:
            # Replace existing lexicon
            await self.save_tenant_lexicon(tenant_id, imported_lexicon)
        
        logger.info(f"Imported {len(imported_lexicon)} entries for tenant {tenant_id}")