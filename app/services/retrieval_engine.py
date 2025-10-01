"""On-Demand Retrieval System - Layer 4 Implementation"""

import re
import logging
import asyncio
import hashlib
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from cachetools import TTLCache
import aiohttp
import json
from urllib.parse import quote
import jellyfish  # For phonetic matching

logger = logging.getLogger(__name__)


class RetrievalDecision(Enum):
    APPLY = "apply"
    SUGGEST = "suggest"
    SKIP = "skip"


@dataclass
class SuspiciousSpan:
    text: str
    segment_idx: int
    position: int
    suspicion_score: float
    indicators: List[str]
    context: str = ""


@dataclass
class RetrievalCandidate:
    text: str
    source: str
    source_type: str  # wikipedia, wikidata, news
    confidence: float
    evidence: str
    url: Optional[str] = None


@dataclass
class RetrievalResult:
    original: str
    candidates: List[RetrievalCandidate]
    consensus: Optional[str]
    decision: RetrievalDecision
    confidence: float
    sources_checked: List[str]


class SuspiciousSpanDetector:
    """Detect suspicious spans that might need correction"""
    
    def __init__(self):
        # Patterns that indicate potential errors
        self.suspicious_patterns = {
            'unusual_title': re.compile(
                r'\b(General|President|Governor|Senator|Minister|Commissioner|Chief)\s+'
                r'(?:The\s+)?[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b'
            ),
            'the_anomaly': re.compile(
                r'\bThe\s+[A-Z][a-z]+\b(?!\s+(of|in|at|on|for|and|or|with))'
            ),
            'mixed_case': re.compile(
                r'\b[a-z]+[A-Z]+[a-z]+\b'
            ),
            'orphaned_initial': re.compile(
                r'\b[A-Z]\.\s+(?![A-Z][a-z])'
            ),
            'unusual_apostrophe': re.compile(
                r"\b[A-Z][a-z]+(?:'[a-z]+)+\b"
            )
        }
        
        # Common ASR error patterns
        self.error_patterns = {
            'number_confusion': re.compile(r'\b(for|to|too)\s+(tree|free)\b', re.IGNORECASE),
            'homophone_error': re.compile(r'\b(there|their|they\'re)\s+(system|name|country)\b', re.IGNORECASE),
        }
        
    def detect_suspicious_spans(
        self,
        segments: List[Dict],
        max_spans: int = 5
    ) -> List[SuspiciousSpan]:
        """Detect suspicious spans in document"""
        suspicious_spans = []
        
        for segment in segments:
            text = segment.get('text', '')
            idx = segment.get('idx', 0)
            
            # Check each pattern
            for pattern_name, pattern in self.suspicious_patterns.items():
                for match in pattern.finditer(text):
                    span_text = match.group(0)
                    score, indicators = self._score_suspicion(span_text, pattern_name)
                    
                    if score > 0.3:  # Threshold for suspicion
                        suspicious_spans.append(SuspiciousSpan(
                            text=span_text,
                            segment_idx=idx,
                            position=match.start(),
                            suspicion_score=score,
                            indicators=indicators,
                            context=text[max(0, match.start()-30):min(len(text), match.end()+30)]
                        ))
        
        # Sort by suspicion score and limit
        suspicious_spans.sort(key=lambda x: x.suspicion_score, reverse=True)
        return suspicious_spans[:max_spans]
    
    def _score_suspicion(self, text: str, pattern_type: str) -> Tuple[float, List[str]]:
        """Score how suspicious a text span is"""
        score = 0.0
        indicators = []
        
        # Base score from pattern type
        pattern_scores = {
            'unusual_title': 0.6,
            'the_anomaly': 0.7,
            'mixed_case': 0.4,
            'orphaned_initial': 0.3,
            'unusual_apostrophe': 0.5
        }
        score = pattern_scores.get(pattern_type, 0.3)
        indicators.append(pattern_type)
        
        # Additional scoring factors
        if 'The ' in text and text.index('The ') > 0:
            score += 0.2
            indicators.append('mid_sentence_the')
        
        # Check for known problem patterns
        if re.search(r'\b(Yahoo|Yahaya|Diyahu)\b', text, re.IGNORECASE):
            score += 0.3
            indicators.append('known_problematic')
        
        # Length factor - very short or very long names are suspicious
        words = text.split()
        if len(words) == 1 and len(text) < 4:
            score += 0.1
            indicators.append('too_short')
        elif len(words) > 4:
            score += 0.1
            indicators.append('too_long')
        
        return min(score, 1.0), indicators


class RegionalContextDetector:
    """Detect regional context from document content"""
    
    def __init__(self):
        # Regional indicators
        self.regional_keywords = {
            'NG': ['nigeria', 'nigerian', 'lagos', 'abuja', 'naira', 'inec', 'efcc'],
            'KE': ['kenya', 'kenyan', 'nairobi', 'mombasa', 'shilling'],
            'GH': ['ghana', 'ghanaian', 'accra', 'kumasi', 'cedi'],
            'ZA': ['south africa', 'johannesburg', 'cape town', 'rand', 'anc'],
            'BO': ['bolivia', 'bolivian', 'la paz', 'santa cruz', 'boliviano'],
            'BR': ['brazil', 'brazilian', 'brasilia', 'sao paulo', 'real'],
            'IN': ['india', 'indian', 'delhi', 'mumbai', 'rupee'],
        }
        
        self.regional_sources = {
            'NG': {
                'news': ['vanguardngr.com', 'punchng.com', 'premiumtimesng.com'],
                'wiki': 'en.wikipedia.org'
            },
            'KE': {
                'news': ['nation.africa', 'standardmedia.co.ke'],
                'wiki': 'en.wikipedia.org'
            },
            'GH': {
                'news': ['graphic.com.gh', 'ghanaweb.com'],
                'wiki': 'en.wikipedia.org'
            },
            'DEFAULT': {
                'news': [],
                'wiki': 'en.wikipedia.org'
            }
        }
    
    def detect_region(self, segments: List[Dict]) -> Optional[str]:
        """Detect regional context from document"""
        region_scores = {}
        
        # Concatenate text for analysis
        full_text = ' '.join(seg.get('text', '') for seg in segments[:20]).lower()
        
        # Score each region
        for region, keywords in self.regional_keywords.items():
            score = sum(1 for keyword in keywords if keyword in full_text)
            if score > 0:
                region_scores[region] = score
        
        # Return highest scoring region
        if region_scores:
            return max(region_scores, key=region_scores.get)
        
        return None
    
    def get_regional_sources(self, region: Optional[str]) -> Dict:
        """Get news sources for region"""
        return self.regional_sources.get(region, self.regional_sources['DEFAULT'])


class OnDemandRetriever:
    """Layer 4: Retrieve and corroborate entity information"""
    
    def __init__(self,
                 max_spans_per_doc: int = 5,
                 min_sources_agreement: int = 2,
                 cache_ttl: int = 3600):
        self.max_spans = max_spans_per_doc
        self.min_agreement = min_sources_agreement
        self.cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        
        self.span_detector = SuspiciousSpanDetector()
        self.region_detector = RegionalContextDetector()
        
        # Confidence thresholds
        self.apply_threshold = 0.92
        self.suggest_threshold = 0.7
        
        # API endpoints
        self.wikipedia_api = "https://en.wikipedia.org/w/api.php"
        self.wikidata_api = "https://www.wikidata.org/w/api.php"
    
    async def process_document(
        self,
        segments: List[Dict],
        region_hint: Optional[str] = None,
        enable_retrieval: bool = True
    ) -> List[RetrievalResult]:
        """Main entry point for Layer 4"""
        
        if not enable_retrieval:
            return []
        
        # Step 1: Detect suspicious spans
        suspicious_spans = self.span_detector.detect_suspicious_spans(segments, self.max_spans)
        
        if not suspicious_spans:
            logger.info("No suspicious spans detected")
            return []
        
        logger.info(f"Found {len(suspicious_spans)} suspicious spans")
        
        # Step 2: Detect regional context
        if not region_hint:
            region_hint = self.region_detector.detect_region(segments)
            if region_hint:
                logger.info(f"Detected region: {region_hint}")
        
        # Step 3: Retrieve candidates in parallel
        tasks = []
        for span in suspicious_spans:
            tasks.append(self._retrieve_and_corroborate(span, region_hint))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Retrieval error: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _retrieve_and_corroborate(
        self,
        span: SuspiciousSpan,
        region: Optional[str]
    ) -> RetrievalResult:
        """Retrieve and corroborate information for a suspicious span"""
        
        # Check cache
        cache_key = f"{span.text}:{region}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Retrieve from multiple sources
        candidates = []
        sources_checked = []
        
        # Wikipedia search
        wiki_candidates = await self._search_wikipedia(span.text)
        candidates.extend(wiki_candidates)
        sources_checked.append('wikipedia')
        
        # Wikidata search
        wikidata_candidates = await self._search_wikidata(span.text)
        candidates.extend(wikidata_candidates)
        sources_checked.append('wikidata')
        
        # Regional news search (if region detected)
        if region:
            regional_sources = self.region_detector.get_regional_sources(region)
            for news_site in regional_sources.get('news', [])[:2]:  # Limit to 2 news sites
                news_candidates = await self._search_news_site(span.text, news_site)
                candidates.extend(news_candidates)
                sources_checked.append(news_site)
        
        # Corroborate and decide
        result = self._corroborate_and_decide(span.text, candidates, sources_checked)
        
        # Cache result
        self.cache[cache_key] = result
        
        return result
    
    async def _search_wikipedia(self, query: str) -> List[RetrievalCandidate]:
        """Search Wikipedia for entity"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'action': 'opensearch',
                    'search': query,
                    'limit': 3,
                    'format': 'json'
                }
                
                async with session.get(
                    self.wikipedia_api,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    
                    candidates = []
                    if len(data) >= 2 and data[1]:  # data[1] contains titles
                        for i, title in enumerate(data[1][:3]):
                            # Calculate similarity
                            similarity = self._calculate_similarity(query, title)
                            
                            candidates.append(RetrievalCandidate(
                                text=title,
                                source='wikipedia',
                                source_type='encyclopedia',
                                confidence=similarity * 0.9,  # Wikipedia has high authority
                                evidence=data[2][i] if len(data) > 2 and i < len(data[2]) else "",
                                url=data[3][i] if len(data) > 3 and i < len(data[3]) else None
                            ))
                    
                    return candidates
                    
        except Exception as e:
            logger.error(f"Wikipedia search error for '{query}': {e}")
            return []
    
    async def _search_wikidata(self, query: str) -> List[RetrievalCandidate]:
        """Search Wikidata for entity"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'action': 'wbsearchentities',
                    'search': query,
                    'language': 'en',
                    'limit': 3,
                    'format': 'json'
                }
                
                async with session.get(
                    self.wikidata_api,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    if response.status != 200:
                        return []
                    
                    data = await response.json()
                    
                    candidates = []
                    for item in data.get('search', []):
                        label = item.get('label', '')
                        description = item.get('description', '')
                        
                        # Calculate similarity
                        similarity = self._calculate_similarity(query, label)
                        
                        candidates.append(RetrievalCandidate(
                            text=label,
                            source='wikidata',
                            source_type='knowledge_base',
                            confidence=similarity * 0.85,  # Slightly lower than Wikipedia
                            evidence=description,
                            url=f"https://www.wikidata.org/wiki/{item.get('id', '')}"
                        ))
                    
                    return candidates
                    
        except Exception as e:
            logger.error(f"Wikidata search error for '{query}': {e}")
            return []
    
    async def _search_news_site(self, query: str, domain: str) -> List[RetrievalCandidate]:
        """Search news site for entity (simplified - would use site-specific search in production)"""
        try:
            # Use DuckDuckGo or Google Custom Search API to search specific site
            # This is a simplified implementation
            search_url = f"https://duckduckgo.com/html/?q=site:{domain}+{quote(query)}"
            
            # In production, would parse search results properly
            # For now, return empty to avoid external dependencies
            return []
            
        except Exception as e:
            logger.error(f"News search error for '{query}' on {domain}: {e}")
            return []
    
    def _calculate_similarity(self, query: str, candidate: str) -> float:
        """Calculate similarity between query and candidate"""
        # Normalize for comparison
        query_norm = query.lower().strip()
        candidate_norm = candidate.lower().strip()
        
        # Exact match
        if query_norm == candidate_norm:
            return 1.0
        
        # Levenshtein distance
        lev_distance = jellyfish.levenshtein_distance(query_norm, candidate_norm)
        max_len = max(len(query_norm), len(candidate_norm))
        lev_similarity = 1 - (lev_distance / max_len) if max_len > 0 else 0
        
        # Phonetic similarity (Metaphone)
        try:
            query_phonetic = jellyfish.metaphone(query_norm)
            candidate_phonetic = jellyfish.metaphone(candidate_norm)
            phonetic_similarity = 1.0 if query_phonetic == candidate_phonetic else 0.5
        except:
            phonetic_similarity = 0.5
        
        # Weighted average
        return (lev_similarity * 0.7) + (phonetic_similarity * 0.3)
    
    def _corroborate_and_decide(
        self,
        original: str,
        candidates: List[RetrievalCandidate],
        sources_checked: List[str]
    ) -> RetrievalResult:
        """Corroborate candidates and make decision"""
        
        if not candidates:
            return RetrievalResult(
                original=original,
                candidates=[],
                consensus=None,
                decision=RetrievalDecision.SKIP,
                confidence=0.0,
                sources_checked=sources_checked
            )
        
        # Group candidates by text
        candidate_groups = {}
        for candidate in candidates:
            key = candidate.text.lower()
            if key not in candidate_groups:
                candidate_groups[key] = []
            candidate_groups[key].append(candidate)
        
        # Find consensus (2+ sources agree)
        consensus_candidate = None
        max_confidence = 0.0
        
        for text_key, group in candidate_groups.items():
            if len(group) >= self.min_agreement:
                # Multiple sources agree
                avg_confidence = sum(c.confidence for c in group) / len(group)
                boost = 1.2  # Boost for multi-source agreement
                final_confidence = min(avg_confidence * boost, 1.0)
                
                if final_confidence > max_confidence:
                    max_confidence = final_confidence
                    consensus_candidate = group[0].text
        
        # If no consensus, take highest confidence single candidate
        if not consensus_candidate and candidates:
            best_candidate = max(candidates, key=lambda c: c.confidence)
            consensus_candidate = best_candidate.text
            max_confidence = best_candidate.confidence
        
        # Decide based on confidence
        if max_confidence >= self.apply_threshold:
            decision = RetrievalDecision.APPLY
        elif max_confidence >= self.suggest_threshold:
            decision = RetrievalDecision.SUGGEST
        else:
            decision = RetrievalDecision.SKIP
        
        return RetrievalResult(
            original=original,
            candidates=candidates[:5],  # Limit candidates
            consensus=consensus_candidate if decision != RetrievalDecision.SKIP else None,
            decision=decision,
            confidence=max_confidence,
            sources_checked=sources_checked
        )