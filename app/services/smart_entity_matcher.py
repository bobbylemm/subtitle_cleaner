"""Smart entity matching for subtitle corrections with improved algorithms"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from difflib import SequenceMatcher
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import jellyfish for phonetic matching
try:
    import jellyfish
    PHONETIC_AVAILABLE = True
except ImportError:
    PHONETIC_AVAILABLE = False
    logger.warning("jellyfish library not available for phonetic matching")


class EntityType(Enum):
    """Types of entities we might encounter"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    TITLE = "title"
    UNKNOWN = "unknown"


@dataclass
class EntityMatch:
    """Represents a potential entity match"""
    original: str
    replacement: str
    confidence: float
    match_type: str
    entity_type: EntityType


class SmartEntityMatcher:
    """
    Intelligent entity matching that avoids common words and uses
    context-aware matching strategies.
    """
    
    def __init__(self):
        # Common words to never replace (expanded list)
        self.stop_words = {
            # Articles and basic words
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'could', 'shall',
            # Prepositions
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up',
            'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'over', 'out',
            # Pronouns
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
            'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what', 'which',
            'who', 'whom', 'whose', 'this', 'that', 'these', 'those',
            # Conjunctions
            'and', 'or', 'but', 'if', 'so', 'yet', 'nor', 'as', 'because', 'since',
            'unless', 'although', 'while', 'whereas',
            # Common verbs
            'go', 'going', 'went', 'come', 'came', 'say', 'said', 'make', 'made',
            'get', 'got', 'know', 'knew', 'think', 'thought', 'take', 'took',
            'see', 'saw', 'want', 'wanted', 'use', 'used', 'find', 'found',
            # Common short words
            'no', 'not', 'now', 'na', 'yes', 'ok', 'okay', 'oh', 'ah', 'uh', 'um',
            # Time-related
            'time', 'day', 'year', 'way', 'man', 'thing', 'world',
        }
        
        # Known ASR misrecognition patterns (from analysis)
        self.known_corrections = {
            "general diyahu": "gentle de yahoo",
            "general at yahoo": "gentle de yahoo",
            "opus odima": "hope uzodinma",
            "hope odidika": "hope uzodinma",
            "opozo nima": "hope uzodinma",
            "opus odima": "hope uzodinma",
        }
        
        # Titles that often precede names
        self.name_titles = {
            'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'professor',
            'president', 'governor', 'senator', 'minister',
            'general', 'captain', 'commander', 'sergeant', 'officer',
            'chief', 'chairman', 'director', 'manager'
        }
        
        # Entity type indicators
        self.entity_indicators = {
            EntityType.PERSON: ['mr', 'mrs', 'ms', 'dr', 'prof', 'said', 'told'],
            EntityType.ORGANIZATION: ['company', 'corporation', 'inc', 'ltd', 'agency', 'department'],
            EntityType.LOCATION: ['state', 'city', 'country', 'street', 'road', 'avenue'],
        }
    
    def find_smart_corrections(
        self,
        document_text: str,
        context_entities: Dict[str, any],
        correction_mode: str = "balanced"
    ) -> Dict[str, str]:
        """
        Find intelligent corrections avoiding common word replacement.
        Now enhanced with document-wide normalization.
        
        Args:
            document_text: The full document text
            context_entities: Dictionary of extracted entities from context
            correction_mode: Mode for contextual correction engine
            
        Returns:
            Dictionary of original text -> corrected text mappings
        """
        corrections = {}
        
        # First, use document-wide normalization to find patterns
        from app.services.document_normalizer import EnhancedDocumentNormalizer
        normalizer = EnhancedDocumentNormalizer()
        
        # Get document-based normalizations with contextual correction engine
        doc_normalizations = normalizer.normalize_document(
            document_text,
            context_entities,
            use_contextual_engine=True,
            correction_mode=correction_mode
        )
        corrections.update(doc_normalizations)
        logger.info(f"Document normalizer found {len(doc_normalizations)} corrections")
        
        # Convert entities to searchable format with metadata
        entity_info = self._analyze_entities(context_entities)
        
        # Extract potential entity mentions from document
        document_entities = self._extract_potential_entities(document_text)
        
        logger.info(f"Found {len(document_entities)} potential entities in document")
        logger.info(f"Matching against {len(entity_info)} context entities")
        
        # Match each document entity against context entities
        for doc_entity in document_entities:
            # Skip if already normalized
            if doc_entity in corrections:
                continue
                
            # Skip if it's a common word
            if self._is_common_word(doc_entity):
                continue
            
            # Find best match from context entities
            best_match = self._find_best_match(doc_entity, entity_info)
            
            if best_match and best_match.confidence > 0.7:
                corrections[doc_entity] = best_match.replacement
                logger.info(f"Match ({best_match.match_type}, {best_match.confidence:.2f}): "
                          f"'{doc_entity}' -> '{best_match.replacement}'")
        
        # Filter out overlapping corrections to prevent duplicates
        filtered_corrections = self._filter_overlapping_corrections(corrections)
        
        return filtered_corrections
    
    def _analyze_entities(self, context_entities: Dict[str, any]) -> List[Tuple[str, EntityType]]:
        """Analyze and categorize context entities"""
        entity_info = []
        
        for entity in context_entities.values():
            text = entity.text
            entity_type = self._classify_entity(text)
            entity_info.append((text, entity_type))
        
        return entity_info
    
    def _filter_overlapping_corrections(self, corrections: Dict[str, str]) -> Dict[str, str]:
        """
        Filter out overlapping corrections to prevent duplicate applications.
        For example, if we have both "Romano" -> "Fabrizio Romano" and 
        "Fabrizio Romano" -> "Fabrizio Romano", we keep only the complete one.
        """
        if not corrections:
            return corrections
        
        filtered = {}
        
        # Sort by length (longest first) to prioritize complete matches
        sorted_items = sorted(corrections.items(), key=lambda x: len(x[0]), reverse=True)
        
        for original, replacement in sorted_items:
            # Check if this correction would cause duplication
            should_add = True
            
            # Don't add if the original is already part of the replacement
            # e.g., don't add "Romano" -> "Fabrizio Romano" if we already have full name
            if original in replacement and original != replacement:
                # Check if we already have a correction for the full phrase
                for existing_orig, existing_repl in filtered.items():
                    if original in existing_orig and existing_repl == replacement:
                        should_add = False
                        break
            
            # Don't add if this is a substring of an already added correction
            for existing_orig in filtered:
                if original != existing_orig:
                    # If original is substring of existing, skip it
                    if original in existing_orig and len(original) < len(existing_orig):
                        should_add = False
                        break
                    # If existing is substring of original, remove existing
                    elif existing_orig in original and len(existing_orig) < len(original):
                        del filtered[existing_orig]
                        break
            
            if should_add:
                # Final check: don't create identity corrections
                if original != replacement:
                    filtered[original] = replacement
        
        logger.info(f"Filtered corrections from {len(corrections)} to {len(filtered)}")
        return filtered
    
    def _classify_entity(self, text: str) -> EntityType:
        """Classify an entity based on its characteristics"""
        text_lower = text.lower()
        
        # Check for organization indicators
        for indicator in self.entity_indicators[EntityType.ORGANIZATION]:
            if indicator in text_lower:
                return EntityType.ORGANIZATION
        
        # Check for location indicators
        for indicator in self.entity_indicators[EntityType.LOCATION]:
            if indicator in text_lower:
                return EntityType.LOCATION
        
        # Check if it starts with a title (likely person)
        first_word = text_lower.split()[0] if text_lower.split() else ""
        if first_word in self.name_titles:
            return EntityType.PERSON
        
        # If multiple capitalized words, likely a person name
        if len(re.findall(r'[A-Z][a-z]+', text)) >= 2:
            return EntityType.PERSON
        
        return EntityType.UNKNOWN
    
    def _extract_potential_entities(self, text: str) -> List[str]:
        """Extract potential entity mentions from text"""
        entities = set()
        
        # Pattern 1: Names with titles (e.g., "General Diyahu")
        title_pattern = re.compile(
            r'\b(' + '|'.join(self.name_titles) + r')\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            re.IGNORECASE
        )
        entities.update(title_pattern.findall(text))
        
        # Pattern 2: Consecutive capitalized words (proper nouns)
        # But require at least 2 words or minimum length
        name_pattern = re.compile(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)+\b')
        potential_names = name_pattern.findall(text)
        
        for name in potential_names:
            # Filter out single short words
            if len(name) >= 4 or len(name.split()) >= 2:
                entities.add(name)
        
        # Pattern 3: Known problem phrases
        for pattern in self.known_corrections.keys():
            if pattern in text.lower():
                # Find the actual casing in the text
                regex = re.compile(re.escape(pattern), re.IGNORECASE)
                matches = regex.findall(text)
                entities.update(matches)
        
        # Pattern 4: Single capitalized words that are long enough and not common
        single_cap = re.compile(r'\b[A-Z][a-z]{4,}\b')
        for word in single_cap.findall(text):
            if word.lower() not in self.stop_words:
                entities.add(word)
        
        return list(entities)
    
    def _is_common_word(self, text: str) -> bool:
        """Check if text is a common word that shouldn't be replaced"""
        # Check single words
        if text.lower() in self.stop_words:
            return True
        
        # Check if all words are common
        words = text.lower().split()
        if all(w in self.stop_words for w in words):
            return True
        
        # Don't replace very short words unless they're known entities
        if len(text) <= 2 and text.lower() not in self.known_corrections:
            return True
        
        return False
    
    def _find_best_match(
        self,
        doc_text: str,
        entity_info: List[Tuple[str, EntityType]]
    ) -> Optional[EntityMatch]:
        """Find the best matching entity from context"""
        doc_lower = doc_text.lower()
        best_match = None
        
        # Strategy 1: Check known corrections first
        if doc_lower in self.known_corrections:
            target = self.known_corrections[doc_lower]
            for entity_text, entity_type in entity_info:
                if entity_text.lower() == target:
                    return EntityMatch(
                        original=doc_text,
                        replacement=entity_text,
                        confidence=1.0,
                        match_type="known_correction",
                        entity_type=entity_type
                    )
        
        # Strategy 2: Phonetic matching for likely names
        if PHONETIC_AVAILABLE and self._looks_like_name(doc_text):
            match = self._phonetic_match_entities(doc_text, entity_info)
            if match and match.confidence > 0.8:
                return match
        
        # Strategy 3: Smart fuzzy matching
        match = self._smart_fuzzy_match(doc_text, entity_info)
        if match and match.confidence > 0.75:
            return match
        
        return best_match
    
    def _looks_like_name(self, text: str) -> bool:
        """Check if text looks like a name"""
        # Has title prefix
        first_word = text.split()[0].lower() if text.split() else ""
        if first_word in self.name_titles:
            return True
        
        # Multiple capitalized words
        if len(re.findall(r'[A-Z][a-z]+', text)) >= 2:
            return True
        
        # Single long capitalized word
        if len(text) >= 5 and text[0].isupper():
            return True
        
        return False
    
    def _phonetic_match_entities(
        self,
        doc_text: str,
        entity_info: List[Tuple[str, EntityType]]
    ) -> Optional[EntityMatch]:
        """Phonetic matching specifically for entities"""
        if not PHONETIC_AVAILABLE:
            return None
        
        doc_lower = doc_text.lower()
        doc_words = doc_lower.split()
        best_match = None
        best_score = 0
        
        for entity_text, entity_type in entity_info:
            entity_lower = entity_text.lower()
            entity_words = entity_lower.split()
            
            # Skip if vastly different lengths
            if abs(len(doc_words) - len(entity_words)) > 1:
                continue
            
            # Skip if entity is too short
            if len(entity_text) < 4:
                continue
            
            scores = []
            
            # Compare whole phrase
            try:
                # Metaphone comparison
                doc_meta = jellyfish.metaphone(doc_lower)
                entity_meta = jellyfish.metaphone(entity_lower)
                if doc_meta and entity_meta and doc_meta == entity_meta:
                    scores.append(1.0)
                
                # Soundex for first word (often most important)
                if doc_words and entity_words:
                    if jellyfish.soundex(doc_words[0]) == jellyfish.soundex(entity_words[0]):
                        scores.append(0.9)
                
                # Jaro-Winkler for overall similarity
                jw_score = jellyfish.jaro_winkler(doc_lower, entity_lower)
                if jw_score > 0.85:
                    scores.append(jw_score)
                
                # NYSIIS comparison
                if jellyfish.nysiis(doc_lower) == jellyfish.nysiis(entity_lower):
                    scores.append(0.85)
            except Exception as e:
                logger.debug(f"Phonetic comparison failed: {e}")
                continue
            
            if scores:
                avg_score = sum(scores) / len(scores)
                # Boost score if same word structure
                if len(doc_words) == len(entity_words):
                    avg_score += 0.05
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_match = EntityMatch(
                        original=doc_text,
                        replacement=entity_text,
                        confidence=min(avg_score, 1.0),
                        match_type="phonetic",
                        entity_type=entity_type
                    )
        
        return best_match if best_score > 0.8 else None
    
    def _smart_fuzzy_match(
        self,
        doc_text: str,
        entity_info: List[Tuple[str, EntityType]]
    ) -> Optional[EntityMatch]:
        """Smart fuzzy matching that considers entity characteristics"""
        doc_lower = doc_text.lower()
        doc_words = doc_lower.split()
        best_match = None
        best_score = 0
        
        for entity_text, entity_type in entity_info:
            entity_lower = entity_text.lower()
            entity_words = entity_lower.split()
            
            # Skip very short entities unless exact match
            if len(entity_text) < 4 and doc_lower != entity_lower:
                continue
            
            # Calculate base similarity
            similarity = SequenceMatcher(None, doc_lower, entity_lower).ratio()
            
            # Apply smart boosting based on patterns
            boost = 0
            
            # Boost if first letters match (common ASR error)
            if doc_lower and entity_lower and doc_lower[0] == entity_lower[0]:
                boost += 0.1
            
            # Boost if same number of words
            if len(doc_words) == len(entity_words):
                boost += 0.05
                
                # Extra boost if first words are similar
                if doc_words and entity_words:
                    first_sim = SequenceMatcher(None, doc_words[0], entity_words[0]).ratio()
                    if first_sim > 0.7:
                        boost += 0.1
            
            # Boost if it's a known entity type and pattern matches
            if entity_type == EntityType.PERSON and self._looks_like_name(doc_text):
                boost += 0.05
            
            # Apply boost
            total_score = min(similarity + boost, 1.0)
            
            # Lower threshold for potential phonetic matches (like Mecano -> Upamecano)
            # Check if it could be a truncation or phonetic match
            threshold = 0.7
            if doc_lower in entity_lower or entity_lower.endswith(doc_lower):
                # Likely a truncation, lower threshold
                threshold = 0.6
            
            if total_score > best_score and total_score > threshold:
                best_score = total_score
                best_match = EntityMatch(
                    original=doc_text,
                    replacement=entity_text,
                    confidence=total_score,
                    match_type="fuzzy",
                    entity_type=entity_type
                )
        
        return best_match