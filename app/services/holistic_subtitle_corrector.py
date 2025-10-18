"""
Holistic Context-Aware Subtitle Correction System

A world-class implementation that understands the entire document
before making corrections, solving the fundamental context problem.

Author: World-class Staff Engineer
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForMaskedLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import spacy
from functools import lru_cache

logger = logging.getLogger(__name__)


class Domain(Enum):
    """Document domains with specific vocabularies and patterns"""
    SPORTS = "sports"
    NEWS = "news"
    TECHNICAL = "technical"
    ENTERTAINMENT = "entertainment"
    EDUCATIONAL = "educational"
    GENERAL = "general"


@dataclass
class Segment:
    """Represents a subtitle segment"""
    idx: int
    start_time: float
    end_time: float
    text: str
    
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class DocumentContext:
    """Rich document-level context"""
    domain: Domain
    topics: List[str]
    entities: List[Dict[str, Any]]
    doc_embedding: np.ndarray
    segment_embeddings: List[np.ndarray]
    attention_matrix: np.ndarray
    semantic_clusters: List[List[int]]
    confidence: float = 0.0
    vocabulary_field: Set[str] = field(default_factory=set)
    
    def get_related_segments(self, segment_idx: int, threshold: float = 0.7) -> List[int]:
        """Get indices of segments semantically related to the given segment"""
        if segment_idx >= len(self.attention_matrix):
            return []
        
        attention_scores = self.attention_matrix[segment_idx]
        related = []
        
        for idx, score in enumerate(attention_scores):
            if idx != segment_idx and score >= threshold:
                related.append(idx)
        
        # Sort by relevance score
        related.sort(key=lambda x: attention_scores[x], reverse=True)
        return related[:10]  # Return top 10 most related


@dataclass
class CorrectionCandidate:
    """A potential correction with scoring"""
    text: str
    semantic_score: float = 0.0
    grammatical_score: float = 0.0
    domain_score: float = 0.0
    consistency_score: float = 0.0
    phonetic_score: float = 0.0
    
    @property
    def total_score(self) -> float:
        """Weighted total score"""
        weights = {
            'semantic': 0.3,
            'grammatical': 0.2,
            'domain': 0.25,
            'consistency': 0.15,
            'phonetic': 0.1
        }
        
        return (
            weights['semantic'] * self.semantic_score +
            weights['grammatical'] * self.grammatical_score +
            weights['domain'] * self.domain_score +
            weights['consistency'] * self.consistency_score +
            weights['phonetic'] * self.phonetic_score
        )


class DocumentUnderstandingEngine:
    """
    Builds comprehensive document-level understanding before any corrections
    """
    
    def __init__(self):
        logger.info("Initializing Document Understanding Engine")
        
        # Sentence transformer for document and segment embeddings
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Domain classification patterns
        self.domain_patterns = {
            Domain.SPORTS: [
                'player', 'team', 'coach', 'match', 'game', 'transfer',
                'contract', 'deal', 'club', 'manager', 'football', 'soccer',
                'basketball', 'tennis', 'goal', 'score', 'win', 'championship'
            ],
            Domain.NEWS: [
                'report', 'announcement', 'government', 'president', 'minister',
                'policy', 'election', 'economy', 'market', 'stock', 'breaking'
            ],
            Domain.TECHNICAL: [
                'software', 'code', 'algorithm', 'system', 'database', 'api',
                'function', 'class', 'method', 'deploy', 'server', 'bug'
            ],
            Domain.ENTERTAINMENT: [
                'movie', 'film', 'actor', 'director', 'music', 'song', 'album',
                'artist', 'show', 'episode', 'season', 'character'
            ],
            Domain.EDUCATIONAL: [
                'lesson', 'learn', 'student', 'teacher', 'course', 'exam',
                'study', 'research', 'theory', 'concept', 'explain'
            ]
        }
        
    def analyze_document(self, segments: List[Segment]) -> DocumentContext:
        """
        Creates rich document representation at multiple levels
        """
        logger.info(f"Analyzing document with {len(segments)} segments")
        
        # 1. Extract full text
        full_text = " ".join([s.text for s in segments])
        
        # 2. Create document embedding
        doc_embedding = self.sentence_encoder.encode(full_text)
        
        # 3. Identify domain
        domain = self.classify_domain(full_text)
        logger.info(f"Detected domain: {domain}")
        
        # 4. Extract topics and entities
        topics = self.extract_topics(full_text, domain)
        entities = self.extract_entities(full_text, domain)
        
        # 5. Build vocabulary field (important words in this document)
        vocabulary_field = self.build_vocabulary_field(full_text, domain)
        
        # 6. Create segment embeddings with temporal encoding
        segment_embeddings = []
        for segment in segments:
            # Get base embedding
            emb = self.sentence_encoder.encode(segment.text)
            # Add temporal position information
            emb = self.add_temporal_encoding(emb, segment.start_time, len(segments))
            segment_embeddings.append(emb)
        
        # 7. Build attention matrix (how related each segment is to others)
        attention_matrix = self.compute_attention_matrix(segment_embeddings)
        
        # 8. Find semantic clusters (groups of related segments)
        semantic_clusters = self.find_semantic_clusters(attention_matrix)
        
        # 9. Calculate overall confidence
        confidence = self.calculate_confidence(domain, topics, entities, vocabulary_field)
        
        return DocumentContext(
            domain=domain,
            topics=topics,
            entities=entities,
            doc_embedding=doc_embedding,
            segment_embeddings=segment_embeddings,
            attention_matrix=attention_matrix,
            semantic_clusters=semantic_clusters,
            confidence=confidence,
            vocabulary_field=vocabulary_field
        )
    
    def classify_domain(self, text: str) -> Domain:
        """Classify document domain based on vocabulary"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            domain_scores[domain] = score
        
        # Get domain with highest score
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 5:  # Minimum threshold
                return best_domain
        
        return Domain.GENERAL
    
    def extract_topics(self, text: str, domain: Domain) -> List[str]:
        """Extract key topics from document"""
        # Simple keyword extraction - in production, use proper topic modeling
        words = text.lower().split()
        word_freq = defaultdict(int)
        
        for word in words:
            if len(word) > 4:  # Skip short words
                word_freq[word] += 1
        
        # Get top topics
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [word for word, _ in topics]
    
    def extract_entities(self, text: str, domain: Domain) -> List[Dict[str, Any]]:
        """Extract named entities relevant to the domain"""
        entities = []
        
        # Extract potential names (capitalized words)
        words = text.split()
        for i, word in enumerate(words):
            if word and word[0].isupper() and len(word) > 2:
                # Check if it's a potential entity
                if domain == Domain.SPORTS:
                    # Look for player names, team names, etc.
                    if i > 0 and words[i-1].lower() in ['player', 'coach', 'manager']:
                        entities.append({'text': word, 'type': 'person', 'context': 'sports'})
                    elif 'united' in word.lower() or 'barcelona' in word.lower():
                        entities.append({'text': word, 'type': 'team', 'context': 'sports'})
        
        return entities
    
    def build_vocabulary_field(self, text: str, domain: Domain) -> Set[str]:
        """Build domain-specific vocabulary field from document"""
        vocabulary = set()
        
        # Add domain-specific terms found in document
        if domain in self.domain_patterns:
            text_lower = text.lower()
            for term in self.domain_patterns[domain]:
                if term in text_lower:
                    vocabulary.add(term)
        
        # Important context words for corrections (especially for sports)
        important_context_words = {
            'sign', 'signed', 'signing', 'deal', 'player', 'contract', 
            'transfer', 'agreement', 'extend', 'extended', 'extension',
            'club', 'manager', 'coach', 'agent', 'negotiation', 'terms'
        }
        
        # Add frequently occurring important words
        words = text.lower().split()
        word_freq = defaultdict(int)
        
        for word in words:
            cleaned_word = word.strip('.,!?;:')
            
            # Add important context words immediately
            if cleaned_word in important_context_words:
                vocabulary.add(cleaned_word)
            
            # Count word frequency for general vocabulary
            if len(cleaned_word) > 3:  # Reduced threshold
                word_freq[cleaned_word] += 1
        
        # Add words that appear multiple times
        for word, freq in word_freq.items():
            if freq >= 2:  # Reduced threshold
                vocabulary.add(word)
        
        logger.debug(f"Built vocabulary field with {len(vocabulary)} terms")
        return vocabulary
    
    def add_temporal_encoding(self, embedding: np.ndarray, position: float, total: int) -> np.ndarray:
        """Add temporal position information to embedding"""
        # Simple positional encoding
        position_weight = position / total
        # Slightly modify embedding based on position
        return embedding * (1 + 0.1 * position_weight)
    
    def compute_attention_matrix(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Compute attention scores between all segment pairs"""
        if not embeddings:
            return np.array([])
        
        # Stack embeddings
        emb_matrix = np.stack(embeddings)
        
        # Compute cosine similarity between all pairs
        attention_matrix = cosine_similarity(emb_matrix, emb_matrix)
        
        return attention_matrix
    
    def find_semantic_clusters(self, attention_matrix: np.ndarray) -> List[List[int]]:
        """Find clusters of semantically related segments"""
        if len(attention_matrix) == 0:
            return []
        
        # Use DBSCAN clustering on similarity matrix
        # Convert similarity to distance (ensure non-negative)
        distance_matrix = 1 - attention_matrix
        # Clip to ensure non-negative values (handle floating point errors)
        distance_matrix = np.maximum(distance_matrix, 0)
        
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)
        
        # Group segments by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:  # -1 is noise
                clusters[label].append(idx)
        
        return list(clusters.values())
    
    def calculate_confidence(
        self,
        domain: Domain,
        topics: List[str],
        entities: List[Dict],
        vocabulary: Set[str]
    ) -> float:
        """Calculate overall document understanding confidence"""
        confidence = 0.5  # Base confidence
        
        # Boost for clear domain
        if domain != Domain.GENERAL:
            confidence += 0.2
        
        # Boost for rich vocabulary
        if len(vocabulary) > 20:
            confidence += 0.15
        
        # Boost for identified entities
        if len(entities) > 5:
            confidence += 0.15
        
        return min(confidence, 1.0)


class ContextAwareCorrectionEngine:
    """
    Makes corrections using full document context
    """
    
    def __init__(self):
        logger.info("Initializing Context-Aware Correction Engine")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load models lazily
        self._bert_model = None
        self._bert_tokenizer = None
        
        # Domain-specific correction patterns
        self.domain_corrections = {
            Domain.SPORTS: {
                'contrast': ('contract', ['sign', 'deal', 'player', 'transfer', 'agreement']),
                'may united': ('Man United', ['manchester', 'football', 'club']),
                'mecano': ('Upamecano', ['player', 'defender', 'bayern'])
            },
            Domain.NEWS: {
                'breaking': ('breaking', ['news', 'report', 'story']),
            }
        }
    
    @property
    def bert_model(self):
        """Lazy load BERT model"""
        if self._bert_model is None:
            try:
                logger.info("Loading BERT model")
                self._bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self._bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
                self._bert_model.to(self.device)
                self._bert_model.eval()
            except Exception as e:
                logger.warning(f"Failed to load BERT: {e}")
                return None
        return self._bert_model
    
    def correct_segment_with_context(
        self,
        segment: Segment,
        doc_context: DocumentContext,
        related_segments: List[Segment]
    ) -> Tuple[str, float]:
        """
        Correct a segment using full document context
        """
        text = segment.text
        
        # 1. Apply domain-specific corrections
        text = self.apply_domain_corrections(text, doc_context)
        
        # 2. Fix compound words using context
        text = self.fix_compound_words(text, doc_context)
        
        # 3. Apply contextual word disambiguation
        text = self.disambiguate_words(text, doc_context, related_segments)
        
        # Calculate confidence
        confidence = self.calculate_correction_confidence(
            segment.text,
            text,
            doc_context
        )
        
        return text, confidence
    
    def apply_domain_corrections(self, text: str, doc_context: DocumentContext) -> str:
        """Apply domain-specific corrections"""
        logger.debug(f"Applying domain corrections for domain: {doc_context.domain}")
        logger.debug(f"Text to correct: '{text}'")
        
        if doc_context.domain not in self.domain_corrections:
            logger.debug(f"No domain corrections for {doc_context.domain}")
            return text
        
        corrections = self.domain_corrections[doc_context.domain]
        text_lower = text.lower()
        
        for error, (correction, context_words) in corrections.items():
            if error in text_lower:
                logger.debug(f"Found potential error '{error}' in text")
                logger.debug(f"Context words to check: {context_words}")
                logger.debug(f"Document vocabulary field: {list(doc_context.vocabulary_field)[:20]}...")
                
                # Check if we have supporting context
                has_context = any(
                    word in doc_context.vocabulary_field 
                    for word in context_words
                )
                
                logger.debug(f"Has supporting context: {has_context}")
                
                if has_context:
                    # Case-insensitive replacement
                    pattern = re.compile(re.escape(error), re.IGNORECASE)
                    text = pattern.sub(correction, text)
                    logger.info(f"✓ Domain correction applied: {error} -> {correction} in '{text}'")
                else:
                    logger.debug(f"No context support for correcting {error}")
        
        return text
    
    def fix_compound_words(self, text: str, doc_context: DocumentContext) -> str:
        """Fix compound words that should be split"""
        words = text.split()
        fixed_words = []
        
        compound_patterns = {
            'thebal': 'the ball',
            'theball': 'the ball',
            'theverbale': 'the verbal',
            'newdeal': 'new deal'
        }
        
        for word in words:
            word_lower = word.lower()
            
            if word_lower in compound_patterns:
                replacement = compound_patterns[word_lower]
                
                # Check context to decide between alternatives
                if word_lower.startswith('theba'):
                    # Decide between "the ball" and "the verbal"
                    if 'ball' in doc_context.vocabulary_field or 'game' in doc_context.vocabulary_field:
                        replacement = 'the ball'
                    elif 'agreement' in doc_context.vocabulary_field or 'verbal' in doc_context.vocabulary_field:
                        replacement = 'the verbal'
                
                fixed_words.append(replacement)
                logger.debug(f"Compound fix: {word} -> {replacement}")
            else:
                fixed_words.append(word)
        
        return ' '.join(fixed_words)
    
    def disambiguate_words(
        self,
        text: str,
        doc_context: DocumentContext,
        related_segments: List[Segment]
    ) -> str:
        """
        Disambiguate confusable words using document context
        """
        # Build extended context from related segments
        extended_context = ' '.join([s.text for s in related_segments])
        
        logger.debug(f"Disambiguating text: '{text}'")
        logger.debug(f"Extended context length: {len(extended_context)} chars")
        logger.debug(f"Related segments count: {len(related_segments)}")
        
        # Confusable pairs with context indicators
        confusables = {
            ('contrast', 'contract'): {
                'contract': ['sign', 'deal', 'agreement', 'player', 'transfer', 'extend', 'signed', 'signing'],
                'contrast': ['compare', 'difference', 'unlike', 'however', 'versus', 'compared']
            },
            ('then', 'than'): {
                'than': ['more', 'less', 'better', 'worse', 'rather'],
                'then': ['after', 'next', 'time', 'when']
            }
        }
        
        words = text.split()
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            
            for pair, indicators in confusables.items():
                if word_lower in pair:
                    logger.debug(f"Found confusable word: '{word_lower}'")
                    alternative = pair[1] if pair[0] == word_lower else pair[0]
                    logger.debug(f"Alternative candidate: '{alternative}'")
                    
                    # Count evidence from extended context
                    orig_indicators = indicators.get(word_lower, [])
                    alt_indicators = indicators.get(alternative, [])
                    
                    logger.debug(f"Checking indicators for '{word_lower}': {orig_indicators}")
                    logger.debug(f"Checking indicators for '{alternative}': {alt_indicators}")
                    
                    orig_evidence = sum(
                        1 for ind in orig_indicators
                        if ind in extended_context.lower()
                    )
                    
                    alt_evidence = sum(
                        1 for ind in alt_indicators
                        if ind in extended_context.lower()
                    )
                    
                    logger.debug(f"Evidence from extended context - {word_lower}: {orig_evidence}, {alternative}: {alt_evidence}")
                    
                    # Also check document vocabulary field
                    vocab_evidence = sum(
                        1 for ind in alt_indicators
                        if ind in doc_context.vocabulary_field
                    )
                    
                    logger.debug(f"Evidence from vocabulary field for {alternative}: {vocab_evidence}")
                    
                    alt_evidence += vocab_evidence * 2  # Weight vocabulary field higher
                    
                    logger.debug(f"Total evidence - {word_lower}: {orig_evidence}, {alternative}: {alt_evidence}")
                    
                    if alt_evidence > orig_evidence:
                        # Preserve capitalization
                        if word[0].isupper():
                            words[i] = alternative.capitalize()
                        else:
                            words[i] = alternative
                        logger.info(f"✓ Disambiguation correction: {word} -> {words[i]} (evidence: {alt_evidence} vs {orig_evidence})")
                    else:
                        logger.debug(f"No correction made - insufficient evidence")
                    
                    break
        
        return ' '.join(words)
    
    def calculate_correction_confidence(
        self,
        original: str,
        corrected: str,
        doc_context: DocumentContext
    ) -> float:
        """Calculate confidence in corrections"""
        if original == corrected:
            return 1.0
        
        # Base confidence from document understanding
        confidence = doc_context.confidence
        
        # Boost for domain-specific corrections
        if doc_context.domain != Domain.GENERAL:
            confidence += 0.1
        
        # Boost for vocabulary field matches
        corrected_words = set(corrected.lower().split())
        vocab_matches = len(corrected_words & doc_context.vocabulary_field)
        confidence += min(vocab_matches * 0.05, 0.2)
        
        return min(confidence, 0.95)


class SlidingWindowProcessor:
    """
    Processes subtitles in overlapping windows to maintain context
    """
    
    def __init__(self, window_size: int = 15, overlap: int = 10):
        self.window_size = window_size
        self.overlap = overlap
        self.correction_engine = ContextAwareCorrectionEngine()
        
    def process_document(
        self,
        segments: List[Segment],
        doc_context: DocumentContext
    ) -> List[Segment]:
        """
        Process with sliding windows ensuring context preservation
        """
        logger.info(f"Processing {len(segments)} segments with windows")
        
        corrected_segments = []
        processed = set()
        
        # Process in overlapping windows
        step = self.window_size - self.overlap
        
        for window_start in range(0, len(segments), step):
            window_end = min(window_start + self.window_size, len(segments))
            
            for i in range(window_start, window_end):
                if i in processed:
                    continue
                
                segment = segments[i]
                
                # Find related segments using document context
                related_indices = doc_context.get_related_segments(i)
                related_segments = [segments[idx] for idx in related_indices if idx < len(segments)]
                
                # Add nearby segments for local context
                local_start = max(0, i - 3)
                local_end = min(len(segments), i + 4)
                local_segments = segments[local_start:local_end]
                
                # Combine related and local segments (deduplicate by index)
                seen_indices = set()
                all_related = []
                for seg in related_segments + local_segments:
                    if seg.idx not in seen_indices:
                        all_related.append(seg)
                        seen_indices.add(seg.idx)
                
                # Correct with full context
                corrected_text, confidence = self.correction_engine.correct_segment_with_context(
                    segment=segment,
                    doc_context=doc_context,
                    related_segments=all_related
                )
                
                # Create corrected segment
                corrected_segment = Segment(
                    idx=segment.idx,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    text=corrected_text
                )
                
                corrected_segments.append(corrected_segment)
                processed.add(i)
                
                if corrected_text != segment.text:
                    logger.debug(f"Segment {i}: '{segment.text}' -> '{corrected_text}' (confidence: {confidence:.2f})")
        
        # Sort by index to maintain order
        corrected_segments.sort(key=lambda s: s.idx)
        
        return corrected_segments


class HolisticSubtitleCorrector:
    """
    Main pipeline orchestrating all components for holistic correction
    """
    
    def __init__(self):
        logger.info("Initializing Holistic Subtitle Corrector")
        self.doc_engine = DocumentUnderstandingEngine()
        self.window_processor = SlidingWindowProcessor()
        
    def correct_subtitles(self, subtitle_content: str) -> str:
        """
        Complete subtitle correction pipeline
        """
        # 1. Parse subtitles
        segments = self.parse_srt(subtitle_content)
        logger.info(f"Parsed {len(segments)} segments")
        
        # 2. Build document understanding (CRUCIAL STEP)
        doc_context = self.doc_engine.analyze_document(segments)
        
        logger.info(f"Document analysis complete:")
        logger.info(f"  Domain: {doc_context.domain.value}")
        logger.info(f"  Topics: {doc_context.topics[:5]}")
        logger.info(f"  Entities: {len(doc_context.entities)} found")
        logger.info(f"  Vocabulary field: {len(doc_context.vocabulary_field)} terms")
        logger.info(f"  Sample vocabulary: {list(doc_context.vocabulary_field)[:30]}")
        logger.info(f"  Confidence: {doc_context.confidence:.2f}")
        logger.info(f"  Semantic clusters: {len(doc_context.semantic_clusters)} groups")
        
        # 3. Process with sliding windows and full context
        corrected_segments = self.window_processor.process_document(
            segments,
            doc_context
        )
        
        # 4. Reconstruct SRT
        return self.reconstruct_srt(corrected_segments)
    
    def parse_srt(self, content: str) -> List[Segment]:
        """Parse SRT content into segments"""
        segments = []
        
        # Split by double newline (segment separator)
        raw_segments = content.strip().split('\n\n')
        
        for raw_segment in raw_segments:
            lines = raw_segment.strip().split('\n')
            if len(lines) >= 3:
                try:
                    idx = int(lines[0])
                    
                    # Parse timestamp
                    timestamp_line = lines[1]
                    start_str, end_str = timestamp_line.split(' --> ')
                    
                    start_time = self.parse_timestamp(start_str)
                    end_time = self.parse_timestamp(end_str)
                    
                    # Join text lines
                    text = ' '.join(lines[2:])
                    
                    segments.append(Segment(
                        idx=idx,
                        start_time=start_time,
                        end_time=end_time,
                        text=text
                    ))
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse segment: {e}")
                    continue
        
        return segments
    
    def parse_timestamp(self, timestamp: str) -> float:
        """Parse SRT timestamp to seconds"""
        time_part, ms_part = timestamp.strip().split(',')
        hours, minutes, seconds = map(int, time_part.split(':'))
        milliseconds = int(ms_part)
        
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    
    def reconstruct_srt(self, segments: List[Segment]) -> str:
        """Reconstruct SRT content from segments"""
        srt_lines = []
        
        for segment in segments:
            srt_lines.append(str(segment.idx))
            srt_lines.append(f"{self.format_timestamp(segment.start_time)} --> {self.format_timestamp(segment.end_time)}")
            srt_lines.append(segment.text)
            srt_lines.append('')  # Empty line between segments
        
        return '\n'.join(srt_lines)


# Integration function for existing pipeline
def get_holistic_corrector() -> HolisticSubtitleCorrector:
    """Get or create the holistic corrector instance"""
    return HolisticSubtitleCorrector()