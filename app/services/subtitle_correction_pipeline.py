"""
Scalable subtitle correction pipeline with staged DAG architecture.
Each stage is pure, idempotent, and cacheable.
"""

import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from functools import lru_cache
import numpy as np
from pathlib import Path
import yaml

from app.services.data_driven_disambiguator import (
    DataDrivenDisambiguator, 
    normalize_text,
    locked_spans,
    is_locked
)

logger = logging.getLogger(__name__)


# ============== Data Contracts ==============
@dataclass
class Segment:
    """Immutable segment representation."""
    id: int
    start_ms: int
    end_ms: int
    text: str
    
    def __hash__(self):
        return hash((self.id, self.text))


@dataclass
class ProcessedSegment:
    """Segment with processing metadata."""
    segment: Segment
    normalized: str
    tokens: List[str]
    guards: List[Tuple[int, int, str]]
    corrections: List[Dict[str, Any]] = field(default_factory=list)
    
    
@dataclass  
class ContextNode:
    """Node in context graph."""
    segment_id: int
    embedding: np.ndarray
    neighbors: List[Tuple[int, float]]  # (neighbor_id, similarity)


@dataclass
class CorrectionCandidate:
    """A potential correction."""
    segment_id: int
    original: str
    replacement: str
    score: float
    source: str
    features: Dict[str, float]


# ============== Stage 1: Normalization ==============
class NormalizationStage:
    """Robust text normalization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.edge_punct = re.compile(r"^[\.,!?;:'\"\(\)\[\]]+|[\.,!?;:'\"\(\)\[\]]+$")
        self.space_pattern = re.compile(r'\s+')
        
    def process(self, segments: List[Segment]) -> List[ProcessedSegment]:
        """Normalize segments."""
        results = []
        
        for seg in segments:
            # NFKC normalization
            normalized = unicodedata.normalize('NFKC', seg.text)
            
            # Collapse spaces
            normalized = self.space_pattern.sub(' ', normalized.strip())
            
            # Standardize quotes and dashes
            normalized = normalized.replace('"', '"').replace('"', '"')
            normalized = normalized.replace(''', "'").replace(''', "'") 
            normalized = normalized.replace('–', '-').replace('—', '-')
            
            # Tokenize
            tokens = self._tokenize(normalized)
            
            # Get guard spans
            guards = locked_spans(normalized)
            
            results.append(ProcessedSegment(
                segment=seg,
                normalized=normalized,
                tokens=tokens,
                guards=guards
            ))
            
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Smart tokenization handling fused words."""
        tokens = []
        
        # Split on spaces first
        for token in text.split():
            # Check for fused tokens (e.g., "can't", "we're")
            if "'" in token and len(token) > 2:
                parts = token.split("'")
                if len(parts) == 2 and parts[1] in ['t', 're', 've', 'll', 'd', 's', 'm']:
                    tokens.extend(parts)
                else:
                    tokens.append(token)
            else:
                # Strip only edge punctuation
                cleaned = self.edge_punct.sub('', token)
                if cleaned:
                    tokens.append(cleaned)
                    
        return tokens


# ============== Stage 2: Context Graph ==============
class ContextGraphStage:
    """Build position-aware context graph with top-k only."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alpha = config.get('alpha', 0.7)  # Blend factor
        self.tau = config.get('tau', 7.0)      # Position decay
        self.k = config.get('k', 8)            # Top-k neighbors
        self.floor = config.get('floor', 0.5)  # Min similarity
        
        # Lazy load embedding model
        self._embedder = None
        
    @property
    def embedder(self):
        """Lazy load multilingual sentence embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            model_name = self.config.get('embedding_model', 
                                        'paraphrase-multilingual-MiniLM-L12-v2')
            self._embedder = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        return self._embedder
    
    def blended_similarity(self, i: int, j: int, embeddings: np.ndarray) -> float:
        """Compute blended similarity: α·cosine + (1−α)·exp(−|Δpos|/τ)."""
        if i == j:
            return 1.0
            
        # Cosine similarity
        emb_i = embeddings[i:i+1]
        emb_j = embeddings[j:j+1]
        cos_sim = float(np.dot(emb_i, emb_j.T) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j)))
        
        # Position decay  
        pos_decay = math.exp(-abs(i - j) / self.tau)
        
        # Blend
        return self.alpha * cos_sim + (1 - self.alpha) * pos_decay
    
    def process(self, segments: List[ProcessedSegment]) -> Dict[int, ContextNode]:
        """Build context graph with top-k neighbors only."""
        texts = [seg.normalized for seg in segments]
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        
        # Build graph with top-k only (not full n×n matrix!)
        graph = {}
        n = len(segments)
        
        for i, seg in enumerate(segments):
            # Compute similarities to other segments
            similarities = []
            
            for j in range(n):
                if i == j:
                    continue
                    
                sim = self.blended_similarity(i, j, embeddings)
                if sim >= self.floor:
                    similarities.append((segments[j].segment.id, sim))
            
            # Keep only top-k
            similarities.sort(key=lambda x: -x[1])
            top_k = similarities[:self.k]
            
            graph[seg.segment.id] = ContextNode(
                segment_id=seg.segment.id,
                embedding=embeddings[i],
                neighbors=top_k
            )
            
        return graph


# ============== Stage 3: Guards ==============  
class GuardStage:
    """Multilingual entity and pattern protection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Lazy load NER
        self._ner = None
        
    @property
    def ner(self):
        """Lazy load multilingual NER."""
        if self._ner is None:
            try:
                import spacy
                # Try multilingual model first
                try:
                    self._ner = spacy.load("xx_ent_wiki_sm")
                except:
                    # Fallback to English
                    self._ner = spacy.load("en_core_web_sm")
                logger.info("Loaded NER model")
            except:
                logger.warning("NER model not available")
        return self._ner
    
    def process(self, segments: List[ProcessedSegment]) -> List[ProcessedSegment]:
        """Add NER-based guards to segments."""
        if not self.ner:
            return segments
            
        for seg in segments:
            # Add NER entities to guards
            doc = self.ner(seg.normalized)
            
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'LOC', 'PRODUCT', 'WORK_OF_ART']:
                    seg.guards.append((ent.start_char, ent.end_char, f"NER_{ent.label_}"))
            
            # Sort and merge overlapping guards
            seg.guards = self._merge_guards(seg.guards)
            
        return segments
    
    def _merge_guards(self, guards: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        """Merge overlapping guard spans."""
        if not guards:
            return []
            
        guards.sort()
        merged = [list(guards[0])]
        
        for g in guards[1:]:
            if g[0] <= merged[-1][1]:
                # Overlapping - extend
                merged[-1][1] = max(merged[-1][1], g[1])
                merged[-1][2] = f"{merged[-1][2]}+{g[2]}"
            else:
                # Non-overlapping - add new
                merged.append(list(g))
                
        return [(s, e, t) for s, e, t in merged]


# ============== Stage 4: Pass-1 Conservative ==============
class Pass1Stage:
    """Conservative cleaning with SymSpell and basic corrections."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_edit_ratio = config.get('pass1_max_edit_ratio', 0.15)
        
        # Initialize SymSpell
        self._symspell = None
        self._init_symspell()
        
    def _init_symspell(self):
        """Initialize SymSpell for spelling correction."""
        try:
            from symspellpy import SymSpell, Verbosity
            
            self._symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            
            # Load dictionary
            dict_path = Path(__file__).parent.parent / "resources" / "frequency_dictionary_en_82_765.txt"
            if dict_path.exists():
                self._symspell.load_dictionary(str(dict_path), term_index=0, count_index=1)
                logger.info("Loaded SymSpell dictionary")
        except Exception as e:
            logger.warning(f"SymSpell initialization failed: {e}")
    
    def process(self, segments: List[ProcessedSegment]) -> Tuple[List[ProcessedSegment], List[CorrectionCandidate]]:
        """Apply conservative corrections."""
        candidates = []
        
        for seg in segments:
            if not self._symspell:
                continue
                
            new_tokens = []
            seg_changed = False
            
            for token in seg.tokens:
                # Skip if in guard
                token_pos = seg.normalized.find(token)
                if token_pos >= 0 and is_locked(seg.guards, token_pos, token_pos + len(token)):
                    new_tokens.append(token)
                    continue
                
                # Try spelling correction
                from symspellpy import Verbosity
                suggestions = self._symspell.lookup(token.lower(), 
                                                   Verbosity.TOP, 
                                                   max_edit_distance=2)
                
                if suggestions and suggestions[0].distance > 0:
                    replacement = suggestions[0].term
                    
                    # Preserve casing
                    if token[0].isupper():
                        replacement = replacement.capitalize()
                    
                    new_tokens.append(replacement)
                    seg_changed = True
                    
                    candidates.append(CorrectionCandidate(
                        segment_id=seg.segment.id,
                        original=token,
                        replacement=replacement,
                        score=1.0 / (1 + suggestions[0].distance),
                        source="symspell",
                        features={'edit_distance': suggestions[0].distance}
                    ))
                else:
                    new_tokens.append(token)
            
            # Update normalized text if changed
            if seg_changed:
                new_text = ' '.join(new_tokens)
                edit_ratio = self._edit_ratio(seg.normalized, new_text)
                
                if edit_ratio <= self.max_edit_ratio:
                    seg.normalized = new_text
                    seg.tokens = new_tokens
                    seg.corrections.extend([c for c in candidates if c.segment_id == seg.segment.id])
        
        return segments, candidates
    
    def _edit_ratio(self, s1: str, s2: str) -> float:
        """Calculate normalized edit distance."""
        from difflib import SequenceMatcher
        return 1 - SequenceMatcher(None, s1, s2).ratio()


# ============== Stage 5: Rebuild ==============
class RebuildStage:
    """Rebuild vocabulary and context from cleaned Pass-1."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_freq = config.get('vocab_min_freq', 2)
        
    def build_vocabulary(self, segments: List[ProcessedSegment]) -> Set[str]:
        """Build clean vocabulary from Pass-1 output."""
        from collections import Counter
        
        word_counts = Counter()
        
        for seg in segments:
            # Use cleaned normalized text
            text = seg.normalized.lower()
            
            # Remove punctuation for vocabulary
            text = re.sub(r'[^\w\s]', ' ', text)
            tokens = text.split()
            
            word_counts.update(tokens)
        
        # Filter by frequency and length
        vocab = {
            word for word, count in word_counts.items() 
            if count >= self.min_freq and 2 < len(word) < 40
        }
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'}
        vocab -= stopwords
        
        logger.info(f"Built vocabulary with {len(vocab)} words")
        return vocab


# ============== Stage 6: Pass-2 Contextual ==============
class Pass2Stage:
    """Contextual disambiguation using data-driven approach."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_edit_ratio = config.get('pass2_max_edit_ratio', 0.30)
        
    def process(self, segments: List[ProcessedSegment], 
               vocabulary: Set[str],
               context_graph: Dict[int, ContextNode]) -> Tuple[List[ProcessedSegment], List[CorrectionCandidate]]:
        """Apply contextual corrections."""
        
        # Initialize data-driven disambiguator
        disambiguator = DataDrivenDisambiguator(vocabulary, self.config)
        
        candidates = []
        
        for seg in segments:
            # Get neighbor segments for context
            node = context_graph.get(seg.segment.id)
            if not node:
                continue
                
            # Build neighbor text
            neighbor_texts = []
            for neighbor_id, _ in node.neighbors[:8]:
                for other_seg in segments:
                    if other_seg.segment.id == neighbor_id:
                        neighbor_texts.append(other_seg.normalized)
                        break
            
            neighbor_text = " ".join(neighbor_texts)
            
            # Disambiguate
            original = seg.normalized
            disambiguated = disambiguator.disambiguate_sentence(
                original, 
                neighbor_text,
                seg.guards
            )
            
            if disambiguated != original:
                seg.normalized = disambiguated
                seg.tokens = disambiguated.split()
                
                # Track correction
                candidates.append(CorrectionCandidate(
                    segment_id=seg.segment.id,
                    original=original,
                    replacement=disambiguated,
                    score=0.8,  # From disambiguator
                    source="contextual",
                    features={'context_support': len(neighbor_texts)}
                ))
        
        return segments, candidates


# ============== Stage 7: Calibrated Selector ==============
class SelectorStage:
    """Select best corrections with calibrated scoring."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pass1_threshold = config.get('pass1_threshold', 0.75)
        self.pass2_threshold = config.get('pass2_threshold', 0.70)
        
    def select(self, candidates: List[CorrectionCandidate], 
              pass_num: int) -> List[CorrectionCandidate]:
        """Select best candidates per segment."""
        
        threshold = self.pass1_threshold if pass_num == 1 else self.pass2_threshold
        
        # Group by segment
        by_segment = {}
        for cand in candidates:
            if cand.segment_id not in by_segment:
                by_segment[cand.segment_id] = []
            by_segment[cand.segment_id].append(cand)
        
        # Select best per segment
        selected = []
        for seg_id, seg_candidates in by_segment.items():
            # Filter by threshold
            valid = [c for c in seg_candidates if c.score >= threshold]
            
            if valid:
                # Sort by score, then minimal edit
                valid.sort(key=lambda c: (-c.score, len(c.replacement) - len(c.original)))
                selected.append(valid[0])
        
        return selected


# ============== Stage 8: Serializer ==============
class SerializerStage:
    """Round-trip safe SRT serialization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def serialize(self, segments: List[ProcessedSegment]) -> str:
        """Serialize to SRT format."""
        lines = []
        
        for seg in segments:
            # Convert timestamps
            start = self._ms_to_srt_time(seg.segment.start_ms)
            end = self._ms_to_srt_time(seg.segment.end_ms)
            
            lines.append(str(seg.segment.id))
            lines.append(f"{start} --> {end}")
            lines.append(seg.normalized)
            lines.append("")
        
        return '\n'.join(lines)
    
    def _ms_to_srt_time(self, ms: int) -> str:
        """Convert milliseconds to SRT timestamp."""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


# ============== Pipeline Orchestrator ==============
class SubtitleCorrectionPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline with configuration."""
        self.config = self._load_config(config_path)
        
        # Initialize stages
        self.normalizer = NormalizationStage(self.config)
        self.context_builder = ContextGraphStage(self.config)
        self.guard_system = GuardStage(self.config)
        self.pass1 = Pass1Stage(self.config)
        self.rebuilder = RebuildStage(self.config)
        self.pass2 = Pass2Stage(self.config)
        self.selector = SelectorStage(self.config)
        self.serializer = SerializerStage(self.config)
        
        logger.info("Pipeline initialized with all stages")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML."""
        default_config = {
            'alpha': 0.7,
            'tau': 7.0,
            'k': 8,
            'floor': 0.5,
            'pass1_max_edit_ratio': 0.15,
            'pass2_max_edit_ratio': 0.30,
            'pass1_threshold': 0.75,
            'pass2_threshold': 0.70,
            'vocab_min_freq': 2,
            'embedding_model': 'paraphrase-multilingual-MiniLM-L12-v2',
            'mlm_model': 'bert-base-multilingual-cased'
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def process(self, srt_content: str) -> str:
        """Process SRT content through the pipeline."""
        
        # Parse SRT
        segments = self._parse_srt(srt_content)
        if not segments:
            return srt_content
        
        logger.info(f"Processing {len(segments)} segments")
        
        # Stage 1: Normalize
        processed = self.normalizer.process(segments)
        
        # Stage 2: Build context graph  
        context_graph = self.context_builder.process(processed)
        
        # Stage 3: Add guards
        processed = self.guard_system.process(processed)
        
        # Stage 4: Pass-1 conservative
        processed, pass1_candidates = self.pass1.process(processed)
        pass1_selected = self.selector.select(pass1_candidates, pass_num=1)
        
        logger.info(f"Pass-1: {len(pass1_selected)} corrections applied")
        
        # Stage 5: Rebuild vocabulary and context
        clean_vocab = self.rebuilder.build_vocabulary(processed)
        context_graph = self.context_builder.process(processed)  # Rebuild on clean text
        
        # Stage 6: Pass-2 contextual
        processed, pass2_candidates = self.pass2.process(processed, clean_vocab, context_graph)
        pass2_selected = self.selector.select(pass2_candidates, pass_num=2)
        
        logger.info(f"Pass-2: {len(pass2_selected)} corrections applied")
        
        # Stage 7: Serialize
        return self.serializer.serialize(processed)
    
    def _parse_srt(self, content: str) -> List[Segment]:
        """Parse SRT content into segments."""
        segments = []
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    idx = int(lines[0])
                    time_parts = lines[1].split(' --> ')
                    if len(time_parts) == 2:
                        start_ms = self._srt_time_to_ms(time_parts[0].strip())
                        end_ms = self._srt_time_to_ms(time_parts[1].strip())
                        text = ' '.join(lines[2:])
                        
                        segments.append(Segment(
                            id=idx,
                            start_ms=start_ms,
                            end_ms=end_ms,
                            text=text
                        ))
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse block: {e}")
                    continue
        
        return segments
    
    def _srt_time_to_ms(self, time_str: str) -> int:
        """Convert SRT timestamp to milliseconds."""
        parts = time_str.replace(',', ':').split(':')
        if len(parts) == 4:
            h, m, s, ms = map(int, parts)
            return h * 3600000 + m * 60000 + s * 1000 + ms
        return 0


# ============== Model Manager (Singleton) ==============
class ModelManager:
    """Singleton manager for model lifecycle."""
    
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, name: str, loader_func):
        """Get or load a model."""
        if name not in self._models:
            logger.info(f"Loading model: {name}")
            self._models[name] = loader_func()
        return self._models[name]
    
    def preload_all(self):
        """Preload all models at app startup."""
        # This should be called in app initialization
        pass


# ============== Export ==============
def get_pipeline(config_path: Optional[str] = None) -> SubtitleCorrectionPipeline:
    """Get or create pipeline instance."""
    return SubtitleCorrectionPipeline(config_path)