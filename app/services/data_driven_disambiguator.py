"""
Data-driven, topic-agnostic disambiguation module.
No hardcoded confusion pairs - learns from document context and language models.
"""

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# ---------- Utilities ----------
EDGE_PUNCT_RE = re.compile(r"^[\s\-\—\–\(\[\{\'\"\»\«]+|[\s\-\—\–\)\]\}\'\"\»\«]+$")
MULTISPACE_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Robust NFKC normalization with edge punct handling."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")  # non-breaking space
    s = MULTISPACE_RE.sub(" ", s.strip())
    return s


def strip_edge_punct(tok: str) -> str:
    """Remove edge punctuation while preserving internal."""
    return EDGE_PUNCT_RE.sub("", tok)


def levenshtein_ratio(a: str, b: str) -> float:
    """Compute normalized Levenshtein distance."""
    import numpy as np
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0 if la == lb else 1.0
    
    dp = np.zeros((la + 1, lb + 1), dtype=int)
    for i in range(la + 1):
        dp[i, 0] = i
    for j in range(lb + 1):
        dp[0, j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + cost)
    dist = dp[la, lb]
    return 1 - dist / max(1, max(la, lb))


# ---------- Guards ----------
NUMERIC_RE = re.compile(r"\b\d+[\d,.]*%?\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{1,2}[\./-]\d{1,2}[\./-]\d{2,4}\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?\b", re.IGNORECASE)
CURRENCY_RE = re.compile(r"[$€£¥₹]\s*\d+[\d,.]*|\d+[\d,.]*\s*(?:USD|EUR|GBP)")


def locked_spans(text: str) -> List[Tuple[int, int, str]]:
    """Identify spans that should not be modified."""
    spans = []
    for rx, label in [
        (NUMERIC_RE, "NUM"),
        (URL_RE, "URL"),
        (DATE_RE, "DATE"),
        (TIME_RE, "TIME"),
        (CURRENCY_RE, "CURR")
    ]:
        for m in rx.finditer(text):
            spans.append((m.start(), m.end(), label))
    
    # Sort and merge overlapping spans
    spans.sort()
    merged = []
    for s in spans:
        if not merged or s[0] > merged[-1][1]:
            merged.append(list(s))
        else:
            merged[-1][1] = max(merged[-1][1], s[1])
    return [(a, b, l) for a, b, l in merged]


def is_locked(spans: List[Tuple[int, int, str]], start: int, end: int) -> bool:
    """Check if a position overlaps with any locked span."""
    for a, b, _ in spans:
        if not (end <= a or start >= b):
            return True
    return False


# ---------- Candidate Generation ----------
class CandidateGenerator:
    """Generate replacement candidates from multiple sources."""
    
    def __init__(self, vocab: Set[str], config: Optional[Dict] = None):
        self.vocab = {w for w in vocab if 2 < len(w) < 40}
        self.config = config or {}
        
        # Optional resources - lazy loaded
        self.jellyfish = None
        self.rf_process = None
        self.rf_fuzz = None
        self.lexicon = set()
        
        # Multilingual MLM - lazy loaded
        self._tok = None
        self._mlm = None
        
        self._init_optional_resources()
    
    def _init_optional_resources(self):
        """Initialize optional dependencies."""
        try:
            import jellyfish
            self.jellyfish = jellyfish
        except ImportError:
            logger.debug("jellyfish not available for phonetic matching")
        
        try:
            from rapidfuzz import process, fuzz
            self.rf_process = process
            self.rf_fuzz = fuzz
        except ImportError:
            logger.debug("rapidfuzz not available for fuzzy matching")
        
        try:
            from wordfreq import top_n_list
            # Multilingual lexicon
            self.lexicon = set()
            for lang, n in [("en", 50000), ("es", 30000), ("fr", 30000), 
                           ("de", 30000), ("it", 20000), ("pt", 20000)]:
                try:
                    self.lexicon.update(top_n_list(lang, n))
                except:
                    pass
        except ImportError:
            logger.debug("wordfreq not available for lexicon")
    
    @lru_cache(maxsize=1)
    def _ensure_mlm(self):
        """Lazy load multilingual BERT model."""
        if self._mlm is None:
            try:
                from transformers import AutoTokenizer, AutoModelForMaskedLM
                import torch
                
                model_name = self.config.get("mlm_model", "bert-base-multilingual-cased")
                self._tok = AutoTokenizer.from_pretrained(model_name)
                self._mlm = AutoModelForMaskedLM.from_pretrained(model_name)
                self._mlm.eval()
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    self._mlm = self._mlm.cuda()
                    
                logger.info(f"Loaded MLM model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load MLM model: {e}")
    
    def lm_topk(self, context: str, sentence: str, target: str, k: int = 8) -> List[str]:
        """Get top-k predictions from masked language model."""
        try:
            import torch
            import torch.nn.functional as F
            
            self._ensure_mlm()
            if self._mlm is None:
                return []
            
            tok = self._tok
            model = self._mlm
            
            # Create masked sentence
            masked = sentence.replace(target, tok.mask_token, 1)
            combined = (context + " " + masked)[-512:]  # Truncate to model max
            
            inputs = tok(combined, return_tensors="pt", truncation=True, max_length=512)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # Find mask position
            mask_positions = (inputs["input_ids"] == tok.mask_token_id).nonzero(as_tuple=False)
            if len(mask_positions) == 0:
                return []
            
            idx = mask_positions[0, 1]
            logp = F.log_softmax(logits[0, idx, :], dim=-1)
            topk = torch.topk(logp, min(k, logp.size(0)))
            ids = topk.indices.cpu().tolist()
            toks = tok.convert_ids_to_tokens(ids)
            
            # Filter subwords and punctuation
            result = []
            for t in toks:
                if t and t.replace("##", "").isalpha() and len(t) > 2:
                    # Handle subword tokens
                    clean = t.replace("##", "").lower()
                    if clean not in result:
                        result.append(clean)
            
            return result[:k]
            
        except Exception as e:
            logger.debug(f"LM prediction failed: {e}")
            return []
    
    def edit_neighbors(self, token: str, limit: int = 12) -> List[str]:
        """Find edit-distance neighbors from vocabulary."""
        base = token.lower()
        cands = []
        
        if self.rf_process and self.rf_fuzz:
            # Use rapidfuzz for efficient fuzzy matching
            matches = self.rf_process.extract(
                base, 
                self.vocab | self.lexicon, 
                scorer=self.rf_fuzz.ratio,
                limit=limit
            )
            cands = [w for w, score, _ in matches if score >= 80]
        else:
            # Fallback: manual edit distance computation
            bucket = [w for w in self.vocab if abs(len(w) - len(base)) <= 2]
            if len(bucket) > 500:
                bucket = bucket[:500]
            scored = [(w, levenshtein_ratio(base, w)) for w in bucket]
            scored.sort(key=lambda x: -x[1])
            cands = [w for w, s in scored[:limit] if s >= 0.6]
        
        return cands
    
    def phonetic_neighbors(self, token: str) -> List[str]:
        """Find phonetically similar words."""
        if not self.jellyfish:
            return []
        
        try:
            code = self.jellyfish.metaphone(token)
            if not code:
                return []
            
            # Check vocabulary for same metaphone
            results = []
            for w in self.vocab:
                if len(results) >= 10:
                    break
                if self.jellyfish.metaphone(w) == code:
                    results.append(w)
            
            return results
        except:
            return []
    
    def from_context_vocab(self, token: str, neighbor_text: str) -> List[str]:
        """Extract frequent tokens from neighbor context."""
        # Tokenize and normalize neighbor text
        toks = [strip_edge_punct(t.lower()) for t in neighbor_text.split()]
        
        # Count frequencies
        freqs: Dict[str, int] = {}
        for t in toks:
            if t and t in (self.vocab | self.lexicon) and len(t) > 2:
                freqs[t] = freqs.get(t, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(freqs.items(), key=lambda x: (-x[1], x[0]))
        return [w for w, _ in sorted_words[:10] if w != token.lower()]
    
    def generate(self, sentence: str, neighbor_text: str, target: str) -> List[str]:
        """Generate all candidate replacements."""
        target_clean = strip_edge_punct(target).lower()
        if not target_clean or len(target_clean) < 3:
            return []
        
        ctx = neighbor_text[-512:]  # Use recent context
        candidates = set()
        
        # Collect from all sources
        sources = [
            ("mlm", lambda: self.lm_topk(ctx, sentence, target, 8)),
            ("edit", lambda: self.edit_neighbors(target_clean, 12)),
            ("phonetic", lambda: self.phonetic_neighbors(target_clean)),
            ("context", lambda: self.from_context_vocab(target_clean, neighbor_text))
        ]
        
        for name, func in sources:
            try:
                for w in func():
                    if w != target_clean and w.isalpha() and 2 < len(w) < 40:
                        candidates.add(w)
            except Exception as e:
                logger.debug(f"Candidate source {name} failed: {e}")
        
        # Return top candidates
        return list(candidates)[:24]


# ---------- Feature Scoring ----------
@dataclass
class Features:
    """Features for ranking candidates."""
    delta_logp: float      # MLM preference for candidate
    edit_ratio: float      # Edit distance from original
    vocab_support: int     # Appears in document vocabulary
    neighbor_support: int  # Frequency in neighbor segments
    is_locked: int        # Overlaps with protected span


def sigmoid(x: float) -> float:
    """Sigmoid activation for probability."""
    return 1 / (1 + math.exp(-x))


class Ranker:
    """Score and rank replacement candidates."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None, threshold: float = 0.70):
        self.w = weights or {
            "bias": -0.4,
            "delta_logp": 0.7,
            "edit_penalty": 1.0,
            "vocab_support": 0.15,
            "neighbor_support": 0.12,
            "locked_penalty": 2.0,
        }
        self.threshold = threshold
        
        # Lazy MLM for delta computation
        self._tok = None
        self._mlm = None
    
    @lru_cache(maxsize=1)
    def _ensure_mlm(self):
        """Lazy load MLM model for scoring."""
        if self._mlm is None:
            try:
                from transformers import AutoTokenizer, AutoModelForMaskedLM
                import torch
                
                model_name = "bert-base-multilingual-cased"
                self._tok = AutoTokenizer.from_pretrained(model_name)
                self._mlm = AutoModelForMaskedLM.from_pretrained(model_name)
                self._mlm.eval()
                
                if torch.cuda.is_available():
                    self._mlm = self._mlm.cuda()
                    
            except Exception as e:
                logger.warning(f"Could not load MLM for ranker: {e}")
    
    def delta_logp(self, context: str, sentence: str, original: str, alt: str) -> float:
        """Compute log probability difference between candidate and original."""
        try:
            import torch
            import torch.nn.functional as F
            
            self._ensure_mlm()
            if self._mlm is None:
                return 0.0
            
            tok = self._tok
            model = self._mlm
            
            # Mask the original token
            masked = sentence.replace(original, tok.mask_token, 1)
            combined = (context + " " + masked)[-512:]
            
            inputs = tok(combined, return_tensors="pt", truncation=True, max_length=512)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # Find mask position
            mask_positions = (inputs["input_ids"] == tok.mask_token_id).nonzero(as_tuple=False)
            if len(mask_positions) == 0:
                return 0.0
            
            idx = mask_positions[0, 1]
            logp = F.log_softmax(logits[0, idx, :], dim=-1)
            
            # Get token IDs
            alt_tokens = tok.tokenize(alt)
            orig_tokens = tok.tokenize(original)
            
            if alt_tokens and orig_tokens:
                alt_id = tok.convert_tokens_to_ids(alt_tokens[0])
                orig_id = tok.convert_tokens_to_ids(orig_tokens[0])
                
                delta = float(logp[alt_id] - logp[orig_id])
                return max(-10, min(10, delta))  # Clip to reasonable range
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Delta logp computation failed: {e}")
            return 0.0
    
    def score(self, features: Features) -> float:
        """Compute probability score for candidate."""
        x = self.w["bias"]
        x += self.w["delta_logp"] * features.delta_logp
        x -= self.w["edit_penalty"] * features.edit_ratio
        x += self.w["vocab_support"] * features.vocab_support
        x += self.w["neighbor_support"] * min(5, features.neighbor_support) * 0.1
        x -= self.w["locked_penalty"] * features.is_locked
        
        return sigmoid(x)


# ---------- Main Disambiguator ----------
class DataDrivenDisambiguator:
    """Topic-agnostic disambiguation using data-driven candidates and ranking."""
    
    def __init__(self, vocab: Set[str], config: Optional[Dict] = None):
        """
        Initialize disambiguator.
        
        Args:
            vocab: Document vocabulary (cleaned from Pass-1)
            config: Configuration dict with:
                - threshold: Acceptance threshold (default 0.70)
                - max_edit_ratio: Max edit distance ratio per sentence (default 0.30)
                - weights: Feature weights for ranker
                - mlm_model: Name of MLM model to use
        """
        self.config = config or {}
        self.gen = CandidateGenerator(vocab, config)
        
        weights = self.config.get("weights")
        threshold = self.config.get("threshold", 0.70)
        self.rank = Ranker(weights=weights, threshold=threshold)
        
        self.max_edit_ratio = self.config.get("max_edit_ratio", 0.30)
        
        # Cache for repeated queries
        self._cache = {}
    
    def disambiguate_sentence(self, sentence: str, neighbor_text: str, 
                            guards: Optional[List[Tuple[int, int, str]]] = None) -> str:
        """
        Disambiguate a single sentence using context.
        
        Args:
            sentence: Sentence to disambiguate
            neighbor_text: Text from neighbor segments for context
            guards: Optional pre-computed locked spans
            
        Returns:
            Disambiguated sentence
        """
        # Normalize input
        text = normalize_text(sentence)
        original = text
        
        # Get locked spans
        if guards is None:
            guards = locked_spans(text)
        
        # Tokenize
        tokens = text.split()
        if not tokens:
            return sentence
        
        # Track changes
        changed = False
        total_edits = 0
        
        # Process each token
        for i, token in enumerate(tokens):
            # Skip if we've exceeded edit budget
            if total_edits > len(tokens) * 0.4:  # Global safety
                break
            
            # Extract core token
            core = strip_edge_punct(token)
            if not core or len(core) < 3:
                continue
            
            # Check cache
            cache_key = (core, neighbor_text[:100])
            if cache_key in self._cache:
                best_alt, best_p = self._cache[cache_key]
            else:
                # Generate candidates
                candidates = self.gen.generate(text, neighbor_text, core)
                if not candidates:
                    self._cache[cache_key] = (core, 0.0)
                    continue
                
                # Rank candidates
                best_alt = core
                best_p = 0.0
                
                for alt in candidates:
                    if alt == core:
                        continue
                    
                    # Check if position is locked
                    token_start = sum(len(t) + 1 for t in tokens[:i])
                    token_end = token_start + len(token)
                    locked = 1 if is_locked(guards, token_start, token_end) else 0
                    
                    # Skip if locked
                    if locked:
                        continue
                    
                    # Compute features
                    delta = self.rank.delta_logp(neighbor_text[-512:], text, core, alt)
                    edit_ratio = 1 - levenshtein_ratio(core, alt)
                    vocab_support = int(alt in self.gen.vocab)
                    neighbor_support = neighbor_text.lower().count(alt)
                    
                    # Score
                    features = Features(delta, edit_ratio, vocab_support, neighbor_support, 0)
                    p = self.rank.score(features)
                    
                    if p > best_p:
                        best_p = p
                        best_alt = alt
                
                # Cache result
                self._cache[cache_key] = (best_alt, best_p)
            
            # Apply change if above threshold
            if best_alt != core and best_p >= self.rank.threshold:
                # Preserve original casing if possible
                if token[0].isupper() and best_alt[0].islower():
                    best_alt = best_alt.capitalize()
                
                # Build new token preserving edge punctuation
                prefix = token[:len(token) - len(token.lstrip())]
                suffix = token[len(token.rstrip()):]
                new_token = prefix + best_alt + suffix
                
                # Check sentence-level edit budget
                new_text = " ".join(tokens[:i] + [new_token] + tokens[i+1:])
                sentence_edit_ratio = 1 - levenshtein_ratio(original, new_text)
                
                if sentence_edit_ratio <= self.max_edit_ratio:
                    tokens[i] = new_token
                    text = new_text
                    changed = True
                    total_edits += 1
                    
                    logger.debug(f"Replaced '{token}' with '{new_token}' (p={best_p:.3f})")
        
        return text if changed else sentence
    
    def disambiguate_document(self, segments: List[Dict], 
                             neighbor_map: Dict[int, List[int]]) -> List[Dict]:
        """
        Disambiguate an entire document.
        
        Args:
            segments: List of segment dicts with 'id', 'text' keys
            neighbor_map: Map of segment ID to neighbor segment IDs
            
        Returns:
            List of disambiguated segments
        """
        results = []
        
        for seg in segments:
            seg_id = seg['id']
            text = seg['text']
            
            # Build neighbor context
            neighbor_ids = neighbor_map.get(seg_id, [])
            neighbor_texts = []
            for nid in neighbor_ids[:8]:  # Top-8 neighbors
                for n in segments:
                    if n['id'] == nid:
                        neighbor_texts.append(n['text'])
                        break
            
            neighbor_text = " ".join(neighbor_texts)
            
            # Disambiguate
            new_text = self.disambiguate_sentence(text, neighbor_text)
            
            results.append({
                **seg,
                'text': new_text,
                'original': text if new_text != text else None
            })
        
        return results


# ---------- Integration Example ----------
if __name__ == "__main__":
    # Example usage
    segments = [
        {"id": 1, "text": "They will sign a contrast tomorrow"},
        {"id": 2, "text": "The player agreed a new deal"},
        {"id": 3, "text": "May United needs strikers"},
    ]
    
    # Build vocabulary from cleaned Pass-1
    vocab = {"contract", "contrast", "deal", "sign", "player", "united", "manchester", "man"}
    
    # Neighbor map (from context stage)
    neighbor_map = {
        1: [2, 3],
        2: [1, 3],
        3: [1, 2]
    }
    
    # Initialize disambiguator
    disambiguator = DataDrivenDisambiguator(vocab, config={"threshold": 0.65})
    
    # Process
    results = disambiguator.disambiguate_document(segments, neighbor_map)
    
    for r in results:
        if r.get('original'):
            print(f"Segment {r['id']}: '{r['original']}' → '{r['text']}'")