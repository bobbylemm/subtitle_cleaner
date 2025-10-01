"""Entity stabilization using phonetic matching and consensus voting"""

import re
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
import jellyfish
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EntityVariant:
    """Represents a variant of an entity found in the document"""
    text: str
    occurrences: int
    indices: List[int]  # Segment indices where it appears
    phonetic_keys: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    

@dataclass 
class EntityCluster:
    """Group of phonetically similar entity variants"""
    variants: List[EntityVariant]
    canonical_form: Optional[str] = None
    cluster_confidence: float = 0.0
    entity_type: str = "UNKNOWN"  # PERSON, ORG, LOC
    

class EntityStabilizer:
    """
    Detects and stabilizes entity variants within a document
    Uses phonetic matching and consensus voting to choose canonical forms
    """
    
    def __init__(self, 
                 min_occurrences: int = 2,
                 similarity_threshold: float = 0.8,
                 min_cluster_size: int = 2):
        """
        Args:
            min_occurrences: Minimum times an entity must appear to be considered
            similarity_threshold: Min similarity score to cluster variants
            min_cluster_size: Minimum variants needed to form a cluster
        """
        self.min_occurrences = min_occurrences
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        
        # Patterns for entity detection
        self.entity_patterns = [
            # Capitalized words (proper nouns)
            re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'),
            # All caps (acronyms)
            re.compile(r'\b[A-Z]{2,}\b'),
            # Mixed case with apostrophe (names like O'Brien)
            re.compile(r"\b[A-Z][a-z]+(?:'[A-Z]?[a-z]+)?\b"),
            # Titles with names
            re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof|President|Governor|Senator)\.?\s+[A-Z][a-z]+\b')
        ]
        
        # Common words to exclude
        self.stopwords = {
            'The', 'This', 'That', 'These', 'Those', 'There', 'Then',
            'But', 'And', 'Or', 'So', 'If', 'When', 'Where', 'What',
            'How', 'Why', 'Who', 'Which', 'While', 'After', 'Before',
            'During', 'Through', 'About', 'Just', 'Now', 'Here', 'Very'
        }
        
    def stabilize_document(self, segments: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Main entry point to stabilize entities in a document
        
        Args:
            segments: List of subtitle segments with 'text' field
            
        Returns:
            Tuple of (stabilized segments, stabilization report)
        """
        # Extract entities
        entities = self._extract_entities(segments)
        
        if not entities:
            return segments, {"clusters": [], "changes": 0}
            
        # Create clusters of similar entities
        clusters = self._create_clusters(entities)
        
        # Choose canonical form for each cluster
        for cluster in clusters:
            cluster.canonical_form = self._select_canonical_form(cluster)
            
        # Apply stabilization
        stabilized_segments, changes = self._apply_stabilization(segments, clusters)
        
        # Generate report
        report = self._generate_report(clusters, changes)
        
        return stabilized_segments, report
        
    def _extract_entities(self, segments: List[Dict]) -> List[EntityVariant]:
        """Extract potential entities from segments"""
        
        entity_counts = Counter()
        entity_indices = defaultdict(list)
        
        for idx, segment in enumerate(segments):
            text = segment.get('text', '')
            
            # Find all potential entities
            found_entities = set()
            for pattern in self.entity_patterns:
                matches = pattern.findall(text)
                for match in matches:
                    # Skip stopwords and very short matches
                    if match not in self.stopwords and len(match) > 1:
                        found_entities.add(match)
                        
            # Count occurrences and track indices
            for entity in found_entities:
                entity_counts[entity] += 1
                entity_indices[entity].append(idx)
                
        # Create EntityVariant objects
        entities = []
        for text, count in entity_counts.items():
            if count >= 1:  # Consider even single occurrences for variant detection
                variant = EntityVariant(
                    text=text,
                    occurrences=count,
                    indices=entity_indices[text],
                    phonetic_keys=self._generate_phonetic_keys(text)
                )
                entities.append(variant)
                
        return entities
        
    def _generate_phonetic_keys(self, text: str) -> Dict[str, str]:
        """Generate multiple phonetic representations"""
        
        keys = {}
        
        # Clean text for phonetic matching
        clean_text = re.sub(r'[^\w\s]', '', text).lower()
        
        try:
            # Metaphone - good for general phonetic matching
            keys['metaphone'] = jellyfish.metaphone(clean_text)
        except:
            keys['metaphone'] = clean_text
            
        try:
            # Soundex - traditional, works well for surnames
            keys['soundex'] = jellyfish.soundex(clean_text)
        except:
            keys['soundex'] = clean_text[:4]
            
        try:
            # NYSIIS - designed for names
            keys['nysiis'] = jellyfish.nysiis(clean_text)
        except:
            keys['nysiis'] = clean_text
            
        try:
            # Match Rating Codex - good for name matching
            keys['mrc'] = jellyfish.match_rating_codex(clean_text)
        except:
            keys['mrc'] = clean_text
            
        return keys
        
    def _create_clusters(self, entities: List[EntityVariant]) -> List[EntityCluster]:
        """Cluster entities based on phonetic similarity"""
        
        clusters = []
        clustered = set()
        
        for i, entity1 in enumerate(entities):
            if i in clustered:
                continue
                
            # Start new cluster
            cluster_variants = [entity1]
            clustered.add(i)
            
            # Find similar entities
            for j, entity2 in enumerate(entities):
                if j <= i or j in clustered:
                    continue
                    
                similarity = self._calculate_similarity(entity1, entity2)
                if similarity >= self.similarity_threshold:
                    cluster_variants.append(entity2)
                    clustered.add(j)
                    
            # Create cluster if it has multiple variants
            if len(cluster_variants) >= self.min_cluster_size:
                cluster = EntityCluster(
                    variants=cluster_variants,
                    entity_type=self._detect_entity_type(cluster_variants[0].text)
                )
                clusters.append(cluster)
                
        return clusters
        
    def _calculate_similarity(self, entity1: EntityVariant, entity2: EntityVariant) -> float:
        """Calculate similarity between two entity variants"""
        
        scores = []
        
        # 1. Phonetic similarity
        phonetic_scores = []
        for key_type in ['metaphone', 'soundex', 'nysiis', 'mrc']:
            if key_type in entity1.phonetic_keys and key_type in entity2.phonetic_keys:
                if entity1.phonetic_keys[key_type] == entity2.phonetic_keys[key_type]:
                    phonetic_scores.append(1.0)
                else:
                    # Partial phonetic match using edit distance
                    dist = jellyfish.levenshtein_distance(
                        entity1.phonetic_keys[key_type],
                        entity2.phonetic_keys[key_type]
                    )
                    max_len = max(
                        len(entity1.phonetic_keys[key_type]),
                        len(entity2.phonetic_keys[key_type])
                    )
                    if max_len > 0:
                        phonetic_scores.append(1.0 - dist / max_len)
                        
        if phonetic_scores:
            scores.append(np.mean(phonetic_scores))
            
        # 2. String similarity (Jaro-Winkler - good for names)
        try:
            jw_score = jellyfish.jaro_winkler_similarity(
                entity1.text.lower(),
                entity2.text.lower()
            )
            scores.append(jw_score)
        except:
            pass
            
        # 3. Edit distance
        edit_dist = jellyfish.levenshtein_distance(entity1.text, entity2.text)
        max_len = max(len(entity1.text), len(entity2.text))
        if max_len > 0:
            edit_score = 1.0 - (edit_dist / max_len)
            scores.append(edit_score)
            
        # 4. Token overlap (for multi-word entities)
        tokens1 = set(entity1.text.lower().split())
        tokens2 = set(entity2.text.lower().split())
        if tokens1 and tokens2:
            overlap = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            if union > 0:
                scores.append(overlap / union)
                
        # Return weighted average
        if not scores:
            return 0.0
            
        # Weight phonetic similarity higher for name matching
        weights = [0.4, 0.3, 0.2, 0.1][:len(scores)]
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else np.mean(scores)
        
    def _select_canonical_form(self, cluster: EntityCluster) -> str:
        """Select the best canonical form from a cluster of variants"""
        
        variants = cluster.variants
        
        # Strategy 1: Majority voting (most frequent)
        frequency_scores = {}
        for variant in variants:
            frequency_scores[variant.text] = variant.occurrences
            
        # Strategy 2: Length preference (longer = more complete)
        length_scores = {}
        for variant in variants:
            # Prefer reasonable length, not too short or too long
            length = len(variant.text)
            if length < 3:
                length_scores[variant.text] = 0.5
            elif length > 50:
                length_scores[variant.text] = 0.7
            else:
                length_scores[variant.text] = min(1.0, length / 20)
                
        # Strategy 3: Capitalization quality
        case_scores = {}
        for variant in variants:
            score = 1.0
            # Proper title case is preferred
            if variant.text.istitle():
                score = 1.0
            elif variant.text.isupper():
                score = 0.8  # All caps is okay but not ideal
            elif variant.text.islower():
                score = 0.6  # All lowercase is worst
            else:
                # Mixed case - check if it follows patterns
                if re.match(r'^[A-Z][a-z]+', variant.text):
                    score = 0.9
                else:
                    score = 0.7
            case_scores[variant.text] = score
            
        # Strategy 4: Position in document (earlier = more likely correct)
        position_scores = {}
        for variant in variants:
            first_occurrence = min(variant.indices) if variant.indices else float('inf')
            # Normalize to 0-1 (earlier is better)
            position_scores[variant.text] = 1.0 / (1.0 + first_occurrence * 0.1)
            
        # Combine scores
        final_scores = {}
        for variant in variants:
            text = variant.text
            
            # Weighted combination
            score = (
                frequency_scores.get(text, 0) * 0.4 +  # Frequency is most important
                case_scores.get(text, 0) * 0.3 +       # Proper casing matters
                length_scores.get(text, 0) * 0.2 +     # Reasonable length
                position_scores.get(text, 0) * 0.1     # Earlier appearance
            )
            
            final_scores[text] = score
            variant.confidence = score
            
        # Select highest scoring variant
        if final_scores:
            canonical = max(final_scores.items(), key=lambda x: x[1])[0]
            cluster.cluster_confidence = final_scores[canonical]
            return canonical
            
        # Fallback to most frequent
        return max(variants, key=lambda v: v.occurrences).text
        
    def _detect_entity_type(self, text: str) -> str:
        """Detect the type of entity (PERSON, ORG, LOC)"""
        
        # Simple heuristics - can be enhanced with NER
        
        # Person indicators
        person_titles = ['Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Senator', 'President', 'Governor']
        person_patterns = [
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+',  # First Last
            r'^[A-Z]\.\s*[A-Z][a-z]+',      # F. Last
        ]
        
        for title in person_titles:
            if title in text:
                return "PERSON"
                
        for pattern in person_patterns:
            if re.match(pattern, text):
                return "PERSON"
                
        # Organization indicators
        org_suffixes = ['Inc', 'Corp', 'LLC', 'Ltd', 'Company', 'Corporation', 'Group']
        for suffix in org_suffixes:
            if suffix in text:
                return "ORG"
                
        # Location indicators
        location_words = ['State', 'City', 'County', 'River', 'Mountain', 'Street', 'Avenue']
        for word in location_words:
            if word in text:
                return "LOC"
                
        # Default to PERSON for single/double word entities
        word_count = len(text.split())
        if word_count <= 2:
            return "PERSON"
        elif word_count <= 4:
            return "ORG"
        else:
            return "LOC"
            
    def _apply_stabilization(self, 
                            segments: List[Dict], 
                            clusters: List[EntityCluster]) -> Tuple[List[Dict], int]:
        """Apply entity stabilization to segments"""
        
        import copy
        stabilized = copy.deepcopy(segments)
        total_changes = 0
        
        for cluster in clusters:
            if not cluster.canonical_form:
                continue
                
            canonical = cluster.canonical_form
            
            # Replace each variant with canonical form
            for variant in cluster.variants:
                if variant.text == canonical:
                    continue  # Skip if already canonical
                    
                # Only apply if confidence is high enough
                if cluster.cluster_confidence < 0.6:
                    continue
                    
                # Replace in all segments where this variant appears
                for idx in variant.indices:
                    if idx < len(stabilized):
                        original = stabilized[idx]['text']
                        
                        # Case-sensitive replacement to preserve context
                        pattern = re.compile(re.escape(variant.text), re.IGNORECASE)
                        modified = pattern.sub(canonical, original)
                        
                        if modified != original:
                            stabilized[idx]['text'] = modified
                            total_changes += 1
                            
        return stabilized, total_changes
        
    def _generate_report(self, clusters: List[EntityCluster], changes: int) -> Dict:
        """Generate stabilization report"""
        
        report = {
            "total_clusters": len(clusters),
            "total_changes": changes,
            "clusters": []
        }
        
        for cluster in clusters:
            cluster_info = {
                "canonical_form": cluster.canonical_form,
                "confidence": cluster.cluster_confidence,
                "entity_type": cluster.entity_type,
                "variants": [
                    {
                        "text": v.text,
                        "occurrences": v.occurrences,
                        "indices": v.indices,
                        "confidence": v.confidence
                    }
                    for v in cluster.variants
                ],
                "total_occurrences": sum(v.occurrences for v in cluster.variants)
            }
            report["clusters"].append(cluster_info)
            
        # Sort clusters by total occurrences
        report["clusters"].sort(key=lambda x: x["total_occurrences"], reverse=True)
        
        return report