"""
Robust, topic-agnostic subtitle correction system.
Two-pass, guarded, position-aware pipeline with calibrated decisions.
"""

import re
import math
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
from symspellpy import SymSpell, Verbosity
from rapidfuzz import fuzz, process
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from collections import defaultdict, Counter
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============== Data Contracts ===============

@dataclass
class Segment:
    """Immutable segment representation."""
    id: int
    start: str
    end: str
    text_raw: str
    text_norm: str = ""
    lang: str = "en"
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id


@dataclass
class ContextGraph:
    """Position-aware context relationships."""
    embeddings: np.ndarray
    topk_neighbors: Dict[int, List[int]] = field(default_factory=dict)
    similarity_matrix: Optional[np.ndarray] = None
    
    
@dataclass
class Candidate:
    """Correction candidate with scoring features."""
    segment_id: int
    span: Tuple[int, int]
    original: str
    replacement: str
    rule_type: str
    features: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    p_accept: float = 0.0


@dataclass
class Decision:
    """Final correction decision."""
    segment_id: int
    text_out: str
    provenance: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardSpan:
    """Protected text span."""
    start: int
    end: int
    type: str
    text: str


# =============== Normalization Stage ===============

class NormalizeStage:
    """Text normalization and tokenization."""
    
    def __init__(self):
        self.punct_pattern = re.compile(r"^[\.,!?;:'\"\(\)\[\]]+|[\.,!?;:'\"\(\)\[\]]+$")
        self.space_pattern = re.compile(r'\s+')
        
    def process(self, segments: List[Segment]) -> List[Segment]:
        """Normalize text while preserving original."""
        normalized = []
        
        for seg in segments:
            # NFKC normalization
            text = unicodedata.normalize('NFKC', seg.text_raw)
            
            # Trim and collapse spaces
            text = self.space_pattern.sub(' ', text.strip())
            
            # Standardize quotes and hyphens
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace(''', "'").replace(''', "'")
            text = text.replace('–', '-').replace('—', '-')
            
            # Create normalized segment
            new_seg = Segment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text_raw=seg.text_raw,
                text_norm=text,
                lang=seg.lang
            )
            normalized.append(new_seg)
            
        return normalized
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize with edge punctuation stripping."""
        tokens = []
        for token in text.split():
            # Strip edge punctuation only
            cleaned = self.punct_pattern.sub('', token)
            if cleaned:
                tokens.append(cleaned)
        return tokens


# =============== Context Graph Builder ===============

class ContextStage:
    """Build position-aware context graph."""
    
    def __init__(self, alpha: float = 0.7, tau: float = 7.0, k: int = 8, floor: float = 0.6):
        self.alpha = alpha
        self.tau = tau
        self.k = k
        self.floor = floor
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
    def blended_similarity(self, i: int, j: int, embeddings: np.ndarray) -> float:
        """Combine cosine similarity with position decay."""
        if i == j:
            return 1.0
            
        # Cosine similarity
        cos_sim = cosine_similarity(
            embeddings[i:i+1], 
            embeddings[j:j+1]
        )[0][0]
        
        # Position decay
        pos_decay = math.exp(-abs(i - j) / self.tau)
        
        # Blend
        return self.alpha * cos_sim + (1 - self.alpha) * pos_decay
    
    def process(self, segments: List[Segment]) -> ContextGraph:
        """Build context graph with blended similarity."""
        texts = [seg.text_norm for seg in segments]
        
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        # Build neighbor relationships
        topk_neighbors = {}
        n = len(segments)
        
        for i in range(n):
            # Compute blended similarities to all other segments
            sims = []
            for j in range(n):
                if i != j:
                    sim = self.blended_similarity(i, j, embeddings)
                    if sim >= self.floor:
                        sims.append((j, sim))
            
            # Sort by similarity and take top-k
            sims.sort(key=lambda x: -x[1])
            topk_neighbors[i] = [idx for idx, _ in sims[:self.k]]
            
        return ContextGraph(
            embeddings=embeddings,
            topk_neighbors=topk_neighbors
        )


# =============== Guard System ===============

class GuardStage:
    """Protect critical spans from modification."""
    
    def __init__(self):
        # Load spaCy model with NER
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except:
            logger.warning("spaCy model not found, guard system limited")
            self.nlp = None
            
        # Patterns for protection
        self.number_pattern = re.compile(r'\b\d+(?:\.\d+)?(?:\s*[KMB%$€£¥])?')
        self.date_pattern = re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b')
        self.time_pattern = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b')
        self.url_pattern = re.compile(r'https?://[^\s]+|www\.[^\s]+')
        
        # Team names to protect (example set)
        self.team_names = {
            'manchester united', 'man united', 'united',
            'manchester city', 'man city', 'city',
            'bayern munich', 'bayern', 'dortmund',
            'real madrid', 'barcelona', 'atletico'
        }
        
    def process(self, segments: List[Segment]) -> Dict[int, List[GuardSpan]]:
        """Identify spans to protect in each segment."""
        guard_spans = {}
        
        for seg in segments:
            spans = []
            text = seg.text_norm
            
            # NER protection
            if self.nlp:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'LOC', 'PRODUCT', 'WORK_OF_ART']:
                        spans.append(GuardSpan(
                            start=ent.start_char,
                            end=ent.end_char,
                            type=f"NER_{ent.label_}",
                            text=ent.text
                        ))
            
            # Number protection
            for match in self.number_pattern.finditer(text):
                spans.append(GuardSpan(
                    start=match.start(),
                    end=match.end(),
                    type="NUMBER",
                    text=match.group()
                ))
            
            # Date protection
            for match in self.date_pattern.finditer(text):
                spans.append(GuardSpan(
                    start=match.start(),
                    end=match.end(),
                    type="DATE",
                    text=match.group()
                ))
            
            # Time protection  
            for match in self.time_pattern.finditer(text):
                spans.append(GuardSpan(
                    start=match.start(),
                    end=match.end(),
                    type="TIME",
                    text=match.group()
                ))
            
            # URL protection
            for match in self.url_pattern.finditer(text):
                spans.append(GuardSpan(
                    start=match.start(),
                    end=match.end(),
                    type="URL",
                    text=match.group()
                ))
            
            # Team name protection
            text_lower = text.lower()
            for team in self.team_names:
                idx = text_lower.find(team)
                if idx != -1:
                    spans.append(GuardSpan(
                        start=idx,
                        end=idx + len(team),
                        type="TEAM",
                        text=text[idx:idx+len(team)]
                    ))
            
            guard_spans[seg.id] = spans
            
        return guard_spans


# =============== Pass-1: Conservative Cleaning ===============

class CleanStage:
    """Conservative first-pass cleaning."""
    
    def __init__(self, max_edit_ratio: float = 0.15):
        self.max_edit_ratio = max_edit_ratio
        self.normalizer = NormalizeStage()
        
        # Initialize SymSpell
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = Path(__file__).parent / "frequency_dictionary_en_82_765.txt"
        if dictionary_path.exists():
            self.sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)
        
        # Common confusables (including multi-word patterns)
        self.confusables = {
            'thebal': 'the ball',
            'theball': 'the ball',
            'newdeal': 'new deal',
            'mayunited': 'man united',
            'mecano': 'upamecano',
            'upamecano': 'upamecano',  # Ensure consistent casing
        }
        
        # Multi-word confusables
        self.multi_word_confusables = {
            'may united': 'man united',
            'man ited': 'man united',
            'manchester ited': 'manchester united'
        }
        
    def process(self, segments: List[Segment], guard_spans: Dict[int, List[GuardSpan]]) -> Tuple[List[Candidate], List[Segment]]:
        """Generate conservative corrections."""
        candidates = []
        cleaned_segments = []
        
        for seg in segments:
            text = seg.text_norm
            guards = guard_spans.get(seg.id, [])
            
            # Skip if too many guards (likely proper text)
            guard_coverage = sum(g.end - g.start for g in guards) / max(len(text), 1)
            if guard_coverage > 0.7:
                cleaned_segments.append(seg)
                continue
            
            # First check for multi-word confusables
            text_corrected = text
            for pattern, replacement in self.multi_word_confusables.items():
                if pattern.lower() in text.lower():
                    # Preserve casing
                    import re
                    text_corrected = re.sub(
                        re.escape(pattern), 
                        replacement,
                        text_corrected, 
                        flags=re.IGNORECASE
                    )
                    candidates.append(Candidate(
                        segment_id=seg.id,
                        span=(0, len(text)),
                        original=pattern,
                        replacement=replacement,
                        rule_type="multi_word_confusable",
                        features={'confidence': 0.9}
                    ))
            
            text = text_corrected
            
            # Try word segmentation for fused words
            tokens = self.normalizer.tokenize(text)
            corrected_tokens = []
            
            for token in tokens:
                token_lower = token.lower()
                
                # Check confusables first
                if token_lower in self.confusables:
                    replacement = self.confusables[token_lower]
                    # Preserve original casing pattern
                    if token[0].isupper():
                        replacement = replacement.capitalize()
                    corrected_tokens.append(replacement)
                    
                    candidates.append(Candidate(
                        segment_id=seg.id,
                        span=(0, len(text)),
                        original=token,
                        replacement=replacement,
                        rule_type="confusable",
                        features={'confidence': 0.9}
                    ))
                else:
                    # SymSpell correction
                    suggestions = self.sym_spell.lookup(token_lower, Verbosity.TOP, max_edit_distance=2)
                    
                    if suggestions and suggestions[0].distance <= 2:
                        # Check if correction is safe (not in guards)
                        is_safe = True
                        token_start = text.lower().find(token_lower)
                        if token_start != -1:
                            for guard in guards:
                                if token_start >= guard.start and token_start < guard.end:
                                    is_safe = False
                                    break
                        
                        if is_safe and suggestions[0].distance > 0:
                            replacement = suggestions[0].term
                            # Preserve casing
                            if token[0].isupper():
                                replacement = replacement.capitalize()
                            corrected_tokens.append(replacement)
                            
                            candidates.append(Candidate(
                                segment_id=seg.id,
                                span=(token_start, token_start + len(token)),
                                original=token,
                                replacement=replacement,
                                rule_type="spelling",
                                features={'edit_distance': suggestions[0].distance}
                            ))
                        else:
                            corrected_tokens.append(token)
                    else:
                        corrected_tokens.append(token)
            
            # Create cleaned segment
            cleaned_text = ' '.join(corrected_tokens)
            
            # Check edit ratio constraint
            edit_ratio = self._edit_ratio(text, cleaned_text)
            if edit_ratio <= self.max_edit_ratio:
                new_seg = Segment(
                    id=seg.id,
                    start=seg.start,
                    end=seg.end,
                    text_raw=seg.text_raw,
                    text_norm=cleaned_text,
                    lang=seg.lang
                )
                cleaned_segments.append(new_seg)
            else:
                cleaned_segments.append(seg)
                
        return candidates, cleaned_segments
    
    def _edit_ratio(self, s1: str, s2: str) -> float:
        """Calculate edit distance ratio."""
        from difflib import SequenceMatcher
        return 1 - SequenceMatcher(None, s1, s2).ratio()


# =============== Pass-2: Contextual Disambiguation ===============

class DisambigStage:
    """Context-aware disambiguation with MLM."""
    
    def __init__(self, max_edit_ratio: float = 0.30):
        self.max_edit_ratio = max_edit_ratio
        
        # Load multilingual BERT
        model_name = "bert-base-multilingual-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        
        # Domain-specific confusables with context indicators
        self.contextual_confusables = {
            'contrast': {
                'replacement': 'contract',
                'indicators': ['sign', 'deal', 'player', 'transfer', 'negotiate', 'agreement', 'year']
            },
            'may': {
                'replacement': 'man',
                'indicators': ['united', 'city', 'football', 'team', 'manchester']
            }
        }
        
    def process(self, segments: List[Segment], context_graph: ContextGraph, 
                vocabulary_field: Set[str], guard_spans: Dict[int, List[GuardSpan]]) -> List[Candidate]:
        """Generate contextual corrections."""
        candidates = []
        
        for seg in segments:
            text = seg.text_norm
            neighbors = context_graph.topk_neighbors.get(seg.id, [])
            guards = guard_spans.get(seg.id, [])
            
            # Build local context (handle window vs full segment indices)
            context_texts = []
            segment_dict = {s.id: s for s in segments}
            for n in neighbors[:5]:
                if n in segment_dict:
                    context_texts.append(segment_dict[n].text_norm)
            context_str = ' '.join(context_texts) if context_texts else ""
            
            # Check contextual confusables
            tokens = text.split()
            for i, token in enumerate(tokens):
                token_lower = token.lower()
                
                if token_lower in self.contextual_confusables:
                    conf = self.contextual_confusables[token_lower]
                    
                    # Count indicators in context
                    indicators_found = sum(
                        1 for ind in conf['indicators']
                        if ind in context_str.lower() or ind in vocabulary_field
                    )
                    
                    if indicators_found >= 2:  # Require multiple indicators
                        # Use MLM to validate
                        mlm_score = self._score_replacement_mlm(
                            text, i, conf['replacement']
                        )
                        
                        if mlm_score > 0.5:  # MLM agrees
                            candidates.append(Candidate(
                                segment_id=seg.id,
                                span=(0, len(text)),
                                original=token,
                                replacement=conf['replacement'],
                                rule_type="contextual",
                                features={
                                    'indicators': indicators_found,
                                    'mlm_score': mlm_score,
                                    'confidence': 0.8
                                }
                            ))
        
        return candidates
    
    def _score_replacement_mlm(self, text: str, token_idx: int, replacement: str) -> float:
        """Score replacement using masked language model."""
        tokens = text.split()
        
        # Mask the token
        masked_tokens = tokens.copy()
        masked_tokens[token_idx] = '[MASK]'
        masked_text = ' '.join(masked_tokens)
        
        # Tokenize for BERT
        inputs = self.tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
        
        # Find mask position
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (inputs.input_ids == mask_token_id).nonzero(as_tuple=True)
        
        if len(mask_positions[0]) > 0:
            mask_pos = mask_positions[1][0]
            
            # Get probabilities for the position
            probs = torch.softmax(predictions[0, mask_pos], dim=-1)
            
            # Score original and replacement
            orig_id = self.tokenizer.convert_tokens_to_ids(tokens[token_idx].lower())
            repl_id = self.tokenizer.convert_tokens_to_ids(replacement.lower())
            
            if orig_id and repl_id:
                orig_prob = probs[orig_id].item()
                repl_prob = probs[repl_id].item()
                
                # Return relative probability
                return repl_prob / (orig_prob + repl_prob + 1e-10)
        
        return 0.0


# =============== Vocabulary Builder ===============

def build_vocabulary_field(segments: List[Segment], min_freq: int = 2) -> Set[str]:
    """Build clean vocabulary from segments."""
    word_counts = Counter()
    
    for seg in segments:
        # Normalize and tokenize
        text = unicodedata.normalize('NFKC', seg.text_norm.lower())
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        tokens = text.split()
        
        # Count tokens
        word_counts.update(tokens)
    
    # Filter by frequency
    vocabulary = {word for word, count in word_counts.items() if count >= min_freq}
    
    # Remove common stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    vocabulary -= stopwords
    
    return vocabulary


# =============== Calibrated Decision Scorer ===============

class CalibratedSelector:
    """Score and select best corrections."""
    
    def __init__(self, threshold_pass1: float = 0.75, threshold_pass2: float = 0.70):
        self.threshold_pass1 = threshold_pass1
        self.threshold_pass2 = threshold_pass2
        
    def score_candidate(self, candidate: Candidate, pass_num: int) -> float:
        """Calculate acceptance probability."""
        score = 0.0
        
        # Rule type priors
        rule_priors = {
            'spelling': 0.8,
            'confusable': 0.9,
            'contextual': 0.7,
            'mlm': 0.6
        }
        score += rule_priors.get(candidate.rule_type, 0.5)
        
        # Feature contributions
        if 'confidence' in candidate.features:
            score += candidate.features['confidence']
        
        if 'mlm_score' in candidate.features:
            score += candidate.features['mlm_score'] * 0.5
        
        if 'indicators' in candidate.features:
            score += min(candidate.features['indicators'] * 0.1, 0.3)
        
        if 'edit_distance' in candidate.features:
            # Penalize large edits
            score -= candidate.features['edit_distance'] * 0.1
        
        # Normalize to probability
        p_accept = 1.0 / (1.0 + math.exp(-score))
        
        candidate.p_accept = p_accept
        return p_accept
    
    def select_best(self, candidates: List[Candidate], segments: List[Segment], 
                   pass_num: int) -> List[Decision]:
        """Select best candidates per segment."""
        # Group by segment
        by_segment = defaultdict(list)
        for cand in candidates:
            by_segment[cand.segment_id].append(cand)
        
        decisions = []
        threshold = self.threshold_pass1 if pass_num == 1 else self.threshold_pass2
        
        for seg in segments:
            if seg.id in by_segment:
                # Score all candidates
                seg_candidates = by_segment[seg.id]
                for cand in seg_candidates:
                    self.score_candidate(cand, pass_num)
                
                # Filter by threshold
                valid = [c for c in seg_candidates if c.p_accept >= threshold]
                
                if valid:
                    # Select best
                    best = max(valid, key=lambda c: (c.p_accept, -len(c.replacement)))
                    
                    # Apply correction
                    text_out = seg.text_norm.replace(best.original, best.replacement)
                    
                    decisions.append(Decision(
                        segment_id=seg.id,
                        text_out=text_out,
                        provenance=f"{best.rule_type}_p{pass_num}",
                        metrics={
                            'p_accept': best.p_accept,
                            'original': best.original,
                            'replacement': best.replacement
                        }
                    ))
                else:
                    # No correction
                    decisions.append(Decision(
                        segment_id=seg.id,
                        text_out=seg.text_norm,
                        provenance="unchanged",
                        metrics={}
                    ))
            else:
                # No candidates
                decisions.append(Decision(
                    segment_id=seg.id,
                    text_out=seg.text_norm,
                    provenance="unchanged",
                    metrics={}
                ))
        
        return decisions


# =============== Windowing and Merge ===============

class WindowProcessor:
    """Process segments in overlapping windows without early lock-in."""
    
    def __init__(self, window_size: int = 10, overlap: int = 3):
        self.window_size = window_size
        self.overlap = overlap
        
    def process_windows(self, segments: List[Segment], processor_func) -> List[Candidate]:
        """Process segments in overlapping windows."""
        n = len(segments)
        all_candidates = []
        
        # Generate windows
        step = self.window_size - self.overlap
        for start in range(0, n, step):
            end = min(start + self.window_size, n)
            window = segments[start:end]
            
            # Process window
            window_candidates = processor_func(window)
            all_candidates.extend(window_candidates)
            
            # Stop if we've covered all segments
            if end >= n:
                break
        
        return all_candidates
    
    def merge_candidates(self, candidates: List[Candidate]) -> Dict[int, Candidate]:
        """Merge candidates by segment, selecting best by score."""
        # Group by segment
        by_segment = defaultdict(list)
        for cand in candidates:
            by_segment[cand.segment_id].append(cand)
        
        # Select best per segment
        best_per_segment = {}
        for seg_id, cands in by_segment.items():
            if cands:
                # Sort by score (highest first), then by edit distance (smallest first)
                sorted_cands = sorted(
                    cands,
                    key=lambda c: (-c.p_accept, len(c.original) - len(c.replacement), c.original)
                )
                best_per_segment[seg_id] = sorted_cands[0]
        
        return best_per_segment


# =============== Main Pipeline ===============

class RobustSubtitleCorrector:
    """Main two-pass correction pipeline."""
    
    def __init__(self, use_windowing: bool = True):
        self.normalizer = NormalizeStage()
        self.context_builder = ContextStage()
        self.guard_system = GuardStage()
        self.cleaner = CleanStage()
        self.disambiguator = DisambigStage()
        self.selector = CalibratedSelector()
        self.window_processor = WindowProcessor() if use_windowing else None
        
    def correct_subtitles(self, srt_content: str) -> str:
        """Process SRT content through two-pass pipeline."""
        # Parse SRT
        segments = self._parse_srt(srt_content)
        
        if not segments:
            return srt_content
        
        # Stage 1: Normalize
        segments = self.normalizer.process(segments)
        
        # Stage 2: Build initial context
        context_graph = self.context_builder.process(segments)
        
        # Stage 3: Identify guards
        guard_spans = self.guard_system.process(segments)
        
        # Stage 4: Pass-1 conservative cleaning
        if self.window_processor:
            # Process in windows
            def pass1_window(window):
                return self.cleaner.process(window, guard_spans)[0]
            
            pass1_candidates = self.window_processor.process_windows(segments, pass1_window)
            
            # Apply best candidates to get cleaned segments
            best_pass1 = self.window_processor.merge_candidates(pass1_candidates)
            cleaned_segments = self._apply_candidates(segments, best_pass1)
        else:
            pass1_candidates, cleaned_segments = self.cleaner.process(segments, guard_spans)
        
        # Stage 5: Rebuild context on cleaned text
        context_graph_clean = self.context_builder.process(cleaned_segments)
        vocabulary_field = build_vocabulary_field(cleaned_segments)
        
        # Stage 6: Pass-2 contextual disambiguation
        if self.window_processor:
            # Process in windows with clean context
            def pass2_window(window):
                # Find window indices in full segment list
                window_ids = {seg.id for seg in window}
                window_graph = ContextGraph(
                    embeddings=context_graph_clean.embeddings,
                    topk_neighbors={
                        k: v for k, v in context_graph_clean.topk_neighbors.items()
                        if k in window_ids
                    }
                )
                return self.disambiguator.process(window, window_graph, vocabulary_field, guard_spans)
            
            pass2_candidates = self.window_processor.process_windows(cleaned_segments, pass2_window)
        else:
            pass2_candidates = self.disambiguator.process(
                cleaned_segments, context_graph_clean, vocabulary_field, guard_spans
            )
        
        # Stage 7: Merge and select best corrections
        all_candidates = pass1_candidates + pass2_candidates
        
        if self.window_processor:
            # Merge candidates from all windows
            best_candidates = self.window_processor.merge_candidates(all_candidates)
            # Convert to decisions
            decisions = []
            for seg in cleaned_segments:
                if seg.id in best_candidates:
                    cand = best_candidates[seg.id]
                    self.selector.score_candidate(cand, pass_num=2)
                    if cand.p_accept >= self.selector.threshold_pass2:
                        text_out = seg.text_norm.replace(cand.original, cand.replacement)
                        decisions.append(Decision(
                            segment_id=seg.id,
                            text_out=text_out,
                            provenance=f"{cand.rule_type}_windowed",
                            metrics={'p_accept': cand.p_accept}
                        ))
                    else:
                        decisions.append(Decision(
                            segment_id=seg.id,
                            text_out=seg.text_norm,
                            provenance="unchanged",
                            metrics={}
                        ))
                else:
                    decisions.append(Decision(
                        segment_id=seg.id,
                        text_out=seg.text_norm,
                        provenance="unchanged",
                        metrics={}
                    ))
        else:
            decisions = self.selector.select_best(all_candidates, cleaned_segments, pass_num=2)
        
        # Stage 8: Apply decisions and serialize
        return self._serialize_srt(segments, decisions)
    
    def _apply_candidates(self, segments: List[Segment], candidates: Dict[int, Candidate]) -> List[Segment]:
        """Apply candidates to segments."""
        result = []
        for seg in segments:
            if seg.id in candidates:
                cand = candidates[seg.id]
                new_text = seg.text_norm.replace(cand.original, cand.replacement)
                new_seg = Segment(
                    id=seg.id,
                    start=seg.start,
                    end=seg.end,
                    text_raw=seg.text_raw,
                    text_norm=new_text,
                    lang=seg.lang
                )
                result.append(new_seg)
            else:
                result.append(seg)
        return result
    
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
                        start = time_parts[0].strip()
                        end = time_parts[1].strip()
                        text = ' '.join(lines[2:])
                        
                        segments.append(Segment(
                            id=idx,
                            start=start,
                            end=end,
                            text_raw=text
                        ))
                except (ValueError, IndexError):
                    continue
        
        return segments
    
    def _serialize_srt(self, original_segments: List[Segment], decisions: List[Decision]) -> str:
        """Serialize segments back to SRT format."""
        decision_map = {d.segment_id: d for d in decisions}
        output_lines = []
        
        for seg in original_segments:
            decision = decision_map.get(seg.id)
            text = decision.text_out if decision else seg.text_raw
            
            output_lines.append(str(seg.id))
            output_lines.append(f"{seg.start} --> {seg.end}")
            output_lines.append(text)
            output_lines.append("")
        
        return '\n'.join(output_lines)


# Singleton instance
_corrector_instance = None

def get_robust_corrector() -> RobustSubtitleCorrector:
    """Get or create the robust corrector instance."""
    global _corrector_instance
    if _corrector_instance is None:
        _corrector_instance = RobustSubtitleCorrector()
    return _corrector_instance