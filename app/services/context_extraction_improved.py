"""Enhanced Context Extraction with Improved Entity Recognition"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set
try:
    from cachetools import TTLCache
except ImportError:
    # Simple fallback cache implementation
    class TTLCache:
        def __init__(self, maxsize, ttl):
            self.cache = {}
            self.ttl = ttl
            
        def __contains__(self, key):
            return key in self.cache
            
        def __getitem__(self, key):
            return self.cache[key]
            
        def __setitem__(self, key, value):
            self.cache[key] = value
            
        def get(self, key, default=None):
            return self.cache.get(key, default)

import aiohttp
import spacy
from trafilatura import extract

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities we can extract"""
    PERSON = "person"
    ORGANIZATION = "org"
    LOCATION = "loc"
    PRODUCT = "product"
    EVENT = "event"
    ROLE = "role"
    CONCEPT = "concept"


class SourceType(Enum):
    """Types of context sources"""
    URL = "url"
    FILE = "file"
    TEXT = "text"


@dataclass
class ContextSource:
    """Represents a source of context information"""
    source_type: SourceType
    content: str  # URL, file path, or raw text
    source_id: str
    language: str = "en"
    authority_score: float = 1.0  # Weight for this source


@dataclass
class ExtractedEntity:
    text: str
    canonical_form: str
    entity_type: EntityType
    confidence: float
    source_id: str
    context: str  # Surrounding text
    position: Optional[int] = None
    frequency: int = 1


class ImprovedContextExtractor:
    """
    Improved Layer 3 Implementation with better compound name extraction
    and context-aware entity filtering
    """
    
    def __init__(self, cache_ttl: int = 900):  # 15 minutes cache
        self.cache_ttl = cache_ttl
        self.cache: TTLCache = TTLCache(maxsize=100, ttl=cache_ttl)
        
        # Initialize spaCy for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not available, using pattern-based extraction only")
            self.nlp = None
        
        # Trafilatura settings for optimal extraction
        self.trafilatura_config = {
            'include_comments': False,
            'include_tables': True,
            'include_links': False,
            'include_formatting': False,
            'include_images': False,
            'no_fallback': False,
            'favor_precision': True,
            'deduplicate': True
        }
        
        # Common words that should NEVER be treated as entities
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
            'now', 'you', 'your', 'yours', 'he', 'she', 'it', 'we', 'they',
            'them', 'their', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'would', 'could', 'ought', 'i', 'me', 'my', 'myself',
            'him', 'his', 'himself', 'her', 'hers', 'herself', 'its', 'itself'
        }
        
        # Words that should not be extracted as single entities but might be part of compound names
        self.common_single_words = {
            'de', 'da', 'at', 'do', 'no', 'se', 'so', 'we', 'me', 'be', 'he',
            'or', 'to', 'in', 'on', 'it', 'is', 'as', 'an', 'if', 'go', 'get',
            'let', 'may', 'new', 'one', 'two', 'way', 'use', 'first', 'last',
            'daily', 'post', 'news', 'auto', 'pilot', 'gentle', 'yahoo'
        }
        
        # Enhanced patterns for entity extraction
        self.entity_patterns = {
            # Quoted entities have highest priority (like 'Gentle De Yahoo')
            'quoted_compound_name': re.compile(
                r'''['"']([A-Z][a-z]+(?:\s+(?:De|Da|Di|Van|Von|Le|La|El|Al|Abu|Ibn|Mac|Mc|O')?(?:\s+[A-Z][a-z]+)+))['"']''',
                re.IGNORECASE
            ),
            
            # Persons with titles
            'person_with_title': re.compile(
                r'\b(?:Dr|Prof|President|Governor|Senator|General|Commander|Captain|Mr|Mrs|Ms|Judge|Rev)\.?\s+[A-Z][a-z]+(?:\s+(?:De|Da|Di|Van|Von|Le|La)?(?:\s+[A-Z][a-z]+)+)',
                re.IGNORECASE
            ),
            
            # Compound names (including simple multi-word names and names with particles)
            'compound_name': re.compile(
                r'\b[A-Z][a-z]+(?:\s+(?:De|Da|Di|Van|Von|Le|La|El|Al|Abu|Ibn|Mac|Mc|O\')?\s*[A-Z][a-z]+)+\b'
            ),
            
            # Organizations (enhanced)
            'organization': re.compile(
                r'\b(?:[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:Inc|Corp|LLC|Ltd|Company|Group|Association|Institute|University|College|Foundation|Agency|Department|Bureau|Office|Center|Centre|Army|Force|Police|Command|Network|ESN|Military)\b'
            ),
            
            # Acronyms (but filter common ones)
            'acronym': re.compile(r'\b[A-Z]{2,}(?:\.[A-Z]{2,})*\b')
        }
        
        # Authority weights for different source types
        self.source_weights = {
            'article': 0.9,     # News articles are highly reliable for current events
            'bio': 1.0,         # Guest bios have highest authority
            'notes': 0.9,       # Show notes are very reliable
            'website': 0.7,     # General website content
            'text': 0.6         # User-provided text
        }
    
    async def extract_context(
        self,
        sources: List[ContextSource],
        language: str = "en"
    ) -> Dict[str, ExtractedEntity]:
        """
        Main entry point for context extraction
        Returns a dictionary of canonical_form -> ExtractedEntity
        """
        all_entities = []
        
        # Process sources in parallel
        tasks = []
        for source in sources:
            if source.source_type == SourceType.URL:
                tasks.append(self._extract_from_url(source))
            elif source.source_type == SourceType.FILE:
                tasks.append(self._extract_from_file(source))
            else:  # TEXT
                tasks.append(self._extract_from_text(source))
        
        source_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all entities
        for i, result in enumerate(source_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to extract from source {sources[i].source_id}: {result}")
                continue
            all_entities.extend(result)
        
        # Build authority-weighted lexicon
        lexicon = self._build_lexicon(all_entities)
        
        # Debug: log all entities before filtering
        logger.debug(f"Before filtering: {len(all_entities)} total entities, {len(lexicon)} unique")
        if len(lexicon) > 0:
            logger.debug(f"Sample entities before filter: {list(lexicon.keys())[:10]}")
        
        # Post-process to remove invalid entities
        lexicon = self._filter_lexicon(lexicon)
        
        # Debug: log after filtering
        logger.debug(f"After filtering: {len(lexicon)} entities remain")
        
        logger.info(f"Extracted {len(lexicon)} unique entities from {len(sources)} sources")
        return lexicon
    
    async def _extract_from_url(self, source: ContextSource) -> List[ExtractedEntity]:
        """Extract entities from URL content"""
        cache_key = f"url:{source.content}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Fetch URL content
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(source.content) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch {source.content}: {response.status}")
                        return []
                    
                    html = await response.text()
            
            # Extract clean text from HTML
            text = extract(html, **self.trafilatura_config)
            if not text:
                logger.warning(f"No text extracted from {source.content}")
                return []
            
            # Detect source type
            source_type = self._detect_source_type(source.content, text)
            authority = self.source_weights.get(source_type, 0.7) * source.authority_score
            
            # Extract entities
            entities = self._extract_entities_from_text(
                text,
                source.source_id,
                authority
            )
            
            self.cache[cache_key] = entities
            return entities
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {source.content}")
            return []
        except Exception as e:
            logger.error(f"Error extracting from URL {source.content}: {e}")
            return []
    
    async def _extract_from_file(self, source: ContextSource) -> List[ExtractedEntity]:
        """Extract entities from file content"""
        cache_key = f"file:{source.source_id}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            file_path = Path(source.content)
            if not file_path.exists():
                logger.warning(f"File not found: {source.content}")
                return []
            
            # Read file
            text = file_path.read_text(encoding='utf-8')
            
            # Detect if it's a bio or notes file
            source_type = 'bio' if 'bio' in file_path.stem.lower() else 'notes'
            authority = self.source_weights.get(source_type, 0.8) * source.authority_score
            
            entities = self._extract_entities_from_text(
                text,
                source.source_id,
                authority
            )
            
            self.cache[cache_key] = entities
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting from file {source.content}: {e}")
            return []
    
    async def _extract_from_text(self, source: ContextSource) -> List[ExtractedEntity]:
        """Extract entities from raw text"""
        authority = self.source_weights.get('text', 0.6) * source.authority_score
        
        return self._extract_entities_from_text(
            source.content,
            source.source_id,
            authority
        )
    
    def _extract_entities_from_text(
        self, 
        text: str, 
        source_id: str,
        authority_score: float
    ) -> List[ExtractedEntity]:
        """Core entity extraction logic with improved compound name handling"""
        entities = []
        seen = set()
        
        # Priority 1: Extract quoted compound names (like 'Gentle De Yahoo')
        for pattern_name, pattern in [
            ('quoted_compound_name', self.entity_patterns['quoted_compound_name'])
        ]:
            for match in pattern.finditer(text):
                entity_text = match.group(1).strip()
                canonical = self._normalize_entity(entity_text)
                
                # Validate entity
                if self._is_valid_entity(canonical):
                    if canonical not in seen:
                        seen.add(canonical)
                        entities.append(ExtractedEntity(
                            text=entity_text,
                            canonical_form=canonical,
                            entity_type=EntityType.PERSON,  # Quoted names are usually people
                            confidence=0.95 * authority_score,  # Very high confidence
                            source_id=source_id,
                            context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
                            position=match.start()
                        ))
        
        # Priority 2: Use spaCy NER if available
        if self.nlp and len(text) < 1000000:  # Limit for performance
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
                    canonical = self._normalize_entity(ent.text)
                    
                    # More strict validation for spaCy entities
                    if self._is_valid_entity(canonical, strict=True):
                        if canonical not in seen:
                            seen.add(canonical)
                            entities.append(ExtractedEntity(
                                text=ent.text,
                                canonical_form=canonical,
                                entity_type=self._map_spacy_label(ent.label_),
                                confidence=0.8 * authority_score,
                                source_id=source_id,
                                context=text[max(0, ent.start_char-50):min(len(text), ent.end_char+50)],
                                position=ent.start_char
                            ))
        
        # Priority 3: Pattern-based extraction for specific patterns
        for pattern_name in ['person_with_title', 'compound_name', 'organization']:
            pattern = self.entity_patterns[pattern_name]
            
            for match in pattern.finditer(text):
                entity_text = match.group(0).strip()
                canonical = self._normalize_entity(entity_text)
                
                # Use less strict validation for compound names
                strict = pattern_name != 'compound_name'
                if self._is_valid_entity(canonical, strict=strict):
                    if canonical not in seen:
                        seen.add(canonical)
                        
                        entity_type = self._get_entity_type_from_pattern(pattern_name)
                        confidence = 0.7 * authority_score
                        
                        entities.append(ExtractedEntity(
                            text=entity_text,
                            canonical_form=canonical,
                            entity_type=entity_type,
                            confidence=confidence,
                            source_id=source_id,
                            context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
                            position=match.start()
                        ))
        
        # Priority 4: Acronyms (with strict filtering)
        acronym_pattern = self.entity_patterns['acronym']
        for match in acronym_pattern.finditer(text):
            entity_text = match.group(0).strip()
            
            # Only accept acronyms that are 3+ chars and not common words
            if len(entity_text) >= 3 and entity_text.lower() not in self.common_single_words:
                canonical = entity_text  # Keep acronyms as-is
                
                if canonical not in seen:
                    seen.add(canonical)
                    entities.append(ExtractedEntity(
                        text=entity_text,
                        canonical_form=canonical,
                        entity_type=EntityType.ORGANIZATION,  # Most acronyms are orgs
                        confidence=0.6 * authority_score,
                        source_id=source_id,
                        context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
                        position=match.start()
                    ))
        
        return entities
    
    def _is_valid_entity(self, entity: str, strict: bool = False) -> bool:
        """
        Validate if a string should be considered an entity
        
        Args:
            entity: The entity text to validate
            strict: If True, apply stricter validation rules
        """
        entity_lower = entity.lower()
        
        # Never accept stop words as entities
        if entity_lower in self.stop_words:
            return False
        
        # Reject very short entities
        if len(entity) < 2:
            return False
        
        # Single word validation
        words = entity.split()
        if len(words) == 1:
            # Reject common single words unless they're proper names
            if entity_lower in self.common_single_words:
                return False
            
            # In strict mode, reject all single words under 4 chars
            if strict and len(entity) < 4:
                return False
        
        # Check if it's mostly non-alphabetic
        if not any(c.isalpha() for c in entity):
            return False
        
        # Reject if it's all lowercase (not a proper noun)
        if strict and entity.islower():
            return False
        
        return True
    
    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text for matching"""
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', text.strip())
        
        # Remove trailing punctuation
        normalized = re.sub(r'[.,;:!?]+$', '', normalized)
        
        # Remove quotes
        normalized = normalized.replace('"', '').replace("'", '').replace("'", '').replace("'", '')
        
        # Handle titles consistently
        normalized = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof)\.', r'\1', normalized)
        
        # Preserve compound name particles
        # Don't lowercase particles like De, Van, etc.
        
        return normalized
    
    def _build_lexicon(self, entities: List[ExtractedEntity]) -> Dict[str, ExtractedEntity]:
        """Build a deduplicated lexicon with highest confidence per entity"""
        lexicon = {}
        
        for entity in entities:
            key = entity.canonical_form.lower()
            
            if key not in lexicon or entity.confidence > lexicon[key].confidence:
                lexicon[key] = entity
        
        return lexicon
    
    def _filter_lexicon(self, lexicon: Dict[str, ExtractedEntity]) -> Dict[str, ExtractedEntity]:
        """Post-process lexicon to remove invalid entries"""
        filtered = {}
        
        for key, entity in lexicon.items():
            # Final validation
            if self._is_valid_entity(entity.canonical_form, strict=False):
                filtered[key] = entity
        
        return filtered
    
    def _detect_source_type(self, url: str, text: str) -> str:
        """Detect the type of source from URL and content"""
        url_lower = url.lower()
        text_lower = text[:1000].lower() if text else ""
        
        # News articles get high priority
        if any(word in url_lower for word in ['news', 'post', 'article', 'story']):
            return 'article'
        elif 'bio' in url_lower or 'about' in url_lower:
            return 'bio'
        elif 'notes' in url_lower or 'episode' in url_lower:
            return 'notes'
        else:
            return 'website'
    
    def _map_spacy_label(self, label: str) -> EntityType:
        """Map spaCy NER labels to our entity types"""
        mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,
            'LOC': EntityType.LOCATION
        }
        return mapping.get(label, EntityType.CONCEPT)
    
    def _get_entity_type_from_pattern(self, pattern_name: str) -> EntityType:
        """Determine entity type from pattern name"""
        if 'person' in pattern_name or 'name' in pattern_name:
            return EntityType.PERSON
        elif 'org' in pattern_name:
            return EntityType.ORGANIZATION
        elif 'loc' in pattern_name:
            return EntityType.LOCATION
        elif 'role' in pattern_name or 'title' in pattern_name:
            return EntityType.ROLE
        else:
            return EntityType.CONCEPT