 Design Specification: Layers 3-6 for Subtitle Caption Enhancement System

  Layer 3: Bring-Your-Context System Design

  Architecture Overview

  ┌─────────────────────────────────────────────────────────┐
  │                  Context Extraction Layer                │
  ├─────────────────────────────────────────────────────────┤
  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
  │  │   URL       │  │   File      │  │   Text      │    │
  │  │  Extractor  │  │  Parser     │  │  Processor  │    │
  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
  │         └────────────┬────┴────────────────┘           │
  │                      ▼                                  │
  │         ┌────────────────────────┐                     │
  │         │   Entity Extractor     │                     │
  │         │  - NER Pipeline        │                     │
  │         │  - Pattern Matching    │                     │
  │         │  - Confidence Scoring  │                     │
  │         └────────────┬───────────┘                     │
  │                      ▼                                  │
  │         ┌────────────────────────┐                     │
  │         │   Lexicon Builder      │                     │
  │         │  - Deduplication       │                     │
  │         │  - Authority Weighting │                     │
  │         │  - Conflict Resolution │                     │
  │         └────────────┬───────────┘                     │
  │                      ▼                                  │
  │         ┌────────────────────────┐                     │
  │         │   Context Cache        │                     │
  │         │  - TTL Management      │                     │
  │         │  - Source Tracking     │                     │
  │         └────────────────────────┘                     │
  └─────────────────────────────────────────────────────────┘

  API Specification

  from typing import List, Dict, Optional, Union
  from dataclasses import dataclass
  from enum import Enum

  class SourceType(Enum):
      URL = "url"
      FILE = "file"
      TEXT = "text"

  class EntityType(Enum):
      PERSON = "person"
      ORGANIZATION = "org"
      LOCATION = "location"
      PRODUCT = "product"
      EVENT = "event"

  @dataclass
  class ContextSource:
      source_type: SourceType
      content: str  # URL, file path, or raw text
      authority_score: float = 1.0  # 0.0 to 1.0
      metadata: Optional[Dict] = None

  @dataclass
  class ExtractedEntity:
      text: str
      canonical_form: str
      entity_type: EntityType
      confidence: float
      source_id: str
      context: str  # Surrounding text
      position: Optional[int] = None

  class ContextExtractor:
      """Main interface for Layer 3"""

      async def extract_context(
          self,
          sources: List[ContextSource],
          language: str = "en"
      ) -> Dict[str, ExtractedEntity]:
          """Extract entities from multiple sources"""
          pass

      def build_lexicon(
          self,
          entities: List[ExtractedEntity],
          existing_glossary: Optional[Dict] = None
      ) -> Dict[str, str]:
          """Build per-job lexicon with conflict resolution"""
          pass

  Implementation Details

  # app/services/context_extraction_v2.py

  import aiohttp
  import asyncio
  from bs4 import BeautifulSoup
  import spacy
  from typing import List, Dict, Set
  import hashlib
  from cachetools import TTLCache
  import trafilatura

  class EnhancedContextExtractor:
      def __init__(self):
          self.nlp = spacy.load("en_core_web_sm")
          self.cache = TTLCache(maxsize=100, ttl=900)  # 15 min cache
          self.extractors = {
              SourceType.URL: self._extract_from_url,
              SourceType.FILE: self._extract_from_file,
              SourceType.TEXT: self._extract_from_text
          }

      async def _extract_from_url(self, url: str) -> str:
          """Extract clean text from URL"""
          cache_key = hashlib.md5(url.encode()).hexdigest()
          if cache_key in self.cache:
              return self.cache[cache_key]

          async with aiohttp.ClientSession() as session:
              async with session.get(url, timeout=5) as response:
                  html = await response.text()

          # Use trafilatura for quality extraction
          text = trafilatura.extract(
              html,
              include_comments=False,
              include_tables=True,
              deduplicate=True,
              favor_precision=True
          )

          self.cache[cache_key] = text
          return text

      def _extract_entities(self, text: str) -> List[ExtractedEntity]:
          """Extract entities using spaCy + patterns"""
          doc = self.nlp(text)
          entities = []

          # Named Entity Recognition
          for ent in doc.ents:
              if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT"]:
                  entities.append(ExtractedEntity(
                      text=ent.text,
                      canonical_form=self._normalize(ent.text),
                      entity_type=self._map_entity_type(ent.label_),
                      confidence=0.8,
                      source_id=hashlib.md5(text.encode()).hexdigest()[:8],
                      context=text[max(0, ent.start_char-50):ent.end_char+50]
                  ))

          # Pattern-based extraction for titles
          title_pattern = r'\b(Dr|Prof|President|Governor|Senator|General|Captain)\.?\s+[A-Z][a-z]+'
          # ... pattern matching logic

          return entities

      def _build_authority_weighted_lexicon(
          self,
          entities: List[ExtractedEntity],
          sources: List[ContextSource]
      ) -> Dict[str, str]:
          """Build lexicon with authority weighting"""
          lexicon = {}
          entity_scores = {}

          for entity in entities:
              key = entity.canonical_form.lower()
              score = entity.confidence * self._get_source_authority(entity.source_id, sources)

              if key not in entity_scores or score > entity_scores[key]:
                  entity_scores[key] = score
                  lexicon[key] = entity.text

          return lexicon

  Database Schema

  -- Context extraction tracking
  CREATE TABLE context_sources (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      job_id UUID NOT NULL,
      source_type VARCHAR(10) NOT NULL,
      source_content TEXT NOT NULL,
      authority_score DECIMAL(3,2) DEFAULT 1.0,
      extracted_at TIMESTAMP DEFAULT NOW(),
      entity_count INTEGER DEFAULT 0,
      cache_key VARCHAR(32),
      INDEX idx_job_id (job_id),
      INDEX idx_cache_key (cache_key)
  );

  CREATE TABLE extracted_entities (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      source_id UUID REFERENCES context_sources(id),
      entity_text VARCHAR(255) NOT NULL,
      canonical_form VARCHAR(255) NOT NULL,
      entity_type VARCHAR(20),
      confidence DECIMAL(3,2),
      context TEXT,
      position INTEGER,
      created_at TIMESTAMP DEFAULT NOW(),
      INDEX idx_canonical (canonical_form),
      INDEX idx_source (source_id)
  );

  Layer 4: On-Demand Retrieval System Design

  Architecture Overview

  ┌─────────────────────────────────────────────────────────┐
  │             Suspicious Span Detection & Retrieval        │
  ├─────────────────────────────────────────────────────────┤
  │  ┌─────────────────────────────────────────────────┐    │
  │  │         Suspicious Span Detector                 │    │
  │  │  - Unusual capitalization patterns              │    │
  │  │  - Low corpus frequency                         │    │
  │  │  - Grammar anomalies                            │    │
  │  │  - Phonetic similarity to known entities        │    │
  │  └──────────────────┬──────────────────────────────┘    │
  │                      ▼                                   │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │         Regional Context Detector                │    │
  │  │  - Extract location hints from document         │    │
  │  │  - Map to regional news sources                 │    │
  │  │  - Prioritize local sources                     │    │
  │  └──────────────────┬──────────────────────────────┘    │
  │                      ▼                                   │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │         Parallel Retrieval Engine                │    │
  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐       │    │
  │  │  │Wikipedia │ │ Wikidata │ │Regional  │       │    │
  │  │  │  API     │ │   API    │ │  News    │       │    │
  │  │  └────┬─────┘ └────┬─────┘ └────┬─────┘       │    │
  │  │       └──────────┬──┴────────────┘              │    │
  │  └──────────────────┼──────────────────────────────┘    │
  │                      ▼                                   │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │         Cross-Source Corroborator                │    │
  │  │  - 2+ source agreement rule                     │    │
  │  │  - Phonetic/string similarity scoring           │    │
  │  │  - Confidence threshold gating                  │    │
  │  └──────────────────┬──────────────────────────────┘    │
  │                      ▼                                   │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │         Decision Engine                          │    │
  │  │  - Apply (confidence > 0.92)                    │    │
  │  │  - Suggest (0.7 < confidence < 0.92)            │    │
  │  │  - Skip (confidence < 0.7)                      │    │
  │  └─────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────┘

  API Specification

  # app/services/retrieval_engine.py

  from dataclasses import dataclass
  from typing import List, Optional, Tuple
  import asyncio
  from enum import Enum

  class RetrievalDecision(Enum):
      APPLY = "apply"
      SUGGEST = "suggest"
      SKIP = "skip"

  @dataclass
  class SuspiciousSpan:
      text: str
      position: int
      suspicion_score: float
      indicators: List[str]

  @dataclass
  class RetrievalCandidate:
      text: str
      source: str
      confidence: float
      evidence: str

  @dataclass
  class RetrievalResult:
      original: str
      candidates: List[RetrievalCandidate]
      consensus: Optional[str]
      decision: RetrievalDecision
      confidence: float

  class OnDemandRetriever:
      def __init__(self, 
                   max_spans_per_doc: int = 5,
                   min_sources_agreement: int = 2,
                   cache_ttl: int = 3600):
          self.max_spans = max_spans_per_doc
          self.min_agreement = min_sources_agreement
          self.cache = TTLCache(maxsize=1000, ttl=cache_ttl)

      async def process_document(
          self,
          segments: List[Dict],
          region_hint: Optional[str] = None
      ) -> List[RetrievalResult]:
          """Main entry point for Layer 4"""

          # Step 1: Detect suspicious spans
          suspicious_spans = self._detect_suspicious_spans(segments)

          # Step 2: Limit to top K most suspicious
          top_spans = sorted(
              suspicious_spans,
              key=lambda x: x.suspicion_score,
              reverse=True
          )[:self.max_spans]

          # Step 3: Detect regional context
          if not region_hint:
              region_hint = self._detect_region(segments)

          # Step 4: Retrieve candidates in parallel
          results = await asyncio.gather(*[
              self._retrieve_and_corroborate(span, region_hint)
              for span in top_spans
          ])

          return results

  Implementation Strategy

  class SuspiciousSpanDetector:
      def __init__(self):
          self.patterns = {
              'unusual_title': r'\b(General|President|Governor)\s+[A-Z]?[a-z]*\s*[A-Z][a-z]*\b',
              'mixed_case': r'\b[a-z]+[A-Z]+[a-z]+\b',
              'the_pattern': r'\bThe\s+[A-Z][a-z]+\b(?!\s+(of|in|at|on|for))',
          }

      def score_span(self, text: str) -> Tuple[float, List[str]]:
          """Score how suspicious a text span is"""
          score = 0.0
          indicators = []

          # Check unusual capitalization
          if self._has_unusual_caps(text):
              score += 0.3
              indicators.append("unusual_capitalization")

          # Check against known entity patterns
          if self._matches_title_pattern(text):
              score += 0.4
              indicators.append("title_pattern")

          # Check corpus frequency (would use real corpus)
          if self._is_low_frequency(text):
              score += 0.3
              indicators.append("low_frequency")

          return score, indicators

  class RegionalRetriever:
      REGIONAL_SOURCES = {
          'NG': {  # Nigeria
              'news': ['vanguardngr.com', 'punchng.com', 'premiumtimesng.com'],
              'wiki': 'en.wikipedia.org'
          },
          'BO': {  # Bolivia  
              'news': ['la-razon.com', 'eldeber.com.bo'],
              'wiki': 'es.wikipedia.org'
          },
          # ... more regions
      }

      async def retrieve_from_region(
          self,
          entity: str,
          region: str
      ) -> List[RetrievalCandidate]:
          """Retrieve from region-specific sources"""
          sources = self.REGIONAL_SOURCES.get(region, self.REGIONAL_SOURCES['NG'])

          tasks = []
          for news_site in sources['news']:
              tasks.append(self._search_news_site(entity, news_site))

          tasks.append(self._search_wikipedia(entity, sources['wiki']))
          tasks.append(self._search_wikidata(entity))

          results = await asyncio.gather(*tasks, return_exceptions=True)

          candidates = []
          for result in results:
              if isinstance(result, Exception):
                  continue
              candidates.extend(result)

          return candidates

  Layer 5: LLM Selector Design

  Architecture Overview

  ┌─────────────────────────────────────────────────────────┐
  │                    LLM Selector Layer                    │
  ├─────────────────────────────────────────────────────────┤
  │  ┌─────────────────────────────────────────────────┐    │
  │  │          Ambiguity Detection                     │    │
  │  │  - Multiple candidates with similar scores      │    │
  │  │  - Confidence in 0.4-0.7 range                  │    │
  │  │  - Conflicting sources                          │    │
  │  └──────────────────┬──────────────────────────────┘    │
  │                      ▼                                   │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │          Schema-Bound Request Builder            │    │
  │  │  - Fixed JSON schema                            │    │
  │  │  - Candidates only (no generation)              │    │
  │  │  - Context window limitation                    │    │
  │  └──────────────────┬──────────────────────────────┘    │
  │                      ▼                                   │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │              LLM Interface                       │    │
  │  │  ┌──────────────┐    ┌──────────────┐          │    │
  │  │  │   OpenAI     │    │  Local LLM   │          │    │
  │  │  │   (Premium)  │    │   (Privacy)  │          │    │
  │  │  └──────┬───────┘    └──────┬───────┘          │    │
  │  │         └────────┬──────────┘                   │    │
  │  └──────────────────┼──────────────────────────────┘    │
  │                      ▼                                   │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │          Response Validator                      │    │
  │  │  - Ensure selection from candidates only        │    │
  │  │  - Validate confidence score                    │    │
  │  │  - Apply edit budget constraints                │    │
  │  └─────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────┘

  API Specification

  # app/services/llm_selector.py

  from typing import List, Optional, Dict
  from pydantic import BaseModel, Field
  import json

  class SelectionRequest(BaseModel):
      """Schema-bound request for LLM"""
      candidates: List[str] = Field(..., max_items=5)
      context: str = Field(..., max_length=200)
      original: str = Field(...)
      instruction: str = Field(
          default="Select the most likely correct form from candidates only. "
                  "Do not generate new text. Return selection and confidence."
      )

  class SelectionResponse(BaseModel):
      """Schema-bound response from LLM"""
      selected: str = Field(..., description="Must be from candidates list")
      confidence: float = Field(..., ge=0.0, le=1.0)
      reasoning: Optional[str] = Field(None, max_length=100)

  class LLMSelector:
      def __init__(self, 
                   provider: str = "local",
                   model: str = "llama-3.2-1b",
                   temperature: float = 0.0):
          self.provider = provider
          self.model = model
          self.temperature = temperature
          self.edit_budget = 5  # Max edits per document
          self.edits_made = 0

      async def select_entity(
          self,
          candidates: List[str],
          context: str,
          original: str
      ) -> Optional[SelectionResponse]:
          """Select best candidate using LLM"""

          # Check edit budget
          if self.edits_made >= self.edit_budget:
              return None

          # Build schema-bound request
          request = SelectionRequest(
              candidates=candidates[:5],  # Limit candidates
              context=context[:200],       # Limit context
              original=original
          )

          # Call LLM based on provider
          if self.provider == "local":
              response = await self._call_local_llm(request)
          elif self.provider == "openai":
              response = await self._call_openai(request)
          else:
              raise ValueError(f"Unknown provider: {self.provider}")

          # Validate response
          if response.selected not in candidates:
              # Fallback to highest confidence candidate
              return None

          self.edits_made += 1
          return response

  Local LLM Implementation

  # app/services/local_llm.py

  import onnxruntime as ort
  from transformers import AutoTokenizer
  import numpy as np

  class LocalLLMSelector:
      """Privacy-preserving local LLM for entity selection"""

      def __init__(self, model_path: str = "models/llama-3.2-1b-onnx"):
          self.session = ort.InferenceSession(f"{model_path}/model.onnx")
          self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

      def select(self, request: SelectionRequest) -> SelectionResponse:
          # Format prompt for selection task
          prompt = self._format_selection_prompt(request)

          # Tokenize
          inputs = self.tokenizer(
              prompt,
              return_tensors="np",
              max_length=256,
              truncation=True
          )

          # Run inference
          outputs = self.session.run(
              None,
              {
                  "input_ids": inputs["input_ids"],
                  "attention_mask": inputs["attention_mask"]
              }
          )

          # Parse structured output
          response_text = self.tokenizer.decode(outputs[0][0])
          return self._parse_response(response_text, request.candidates)

      def _format_selection_prompt(self, request: SelectionRequest) -> str:
          return f"""Task: Select the correct entity from candidates.
  Context: {request.context}
  Original: {request.original}
  Candidates: {json.dumps(request.candidates)}
  Select one candidate. Output JSON with 'selected' and 'confidence' fields.
  Response:"""

  Layer 6: Per-Tenant Memory System Design

  Architecture Overview

  ┌─────────────────────────────────────────────────────────┐
  │                 Per-Tenant Memory Layer                  │
  ├─────────────────────────────────────────────────────────┤
  │  ┌─────────────────────────────────────────────────┐    │
  │  │           Tenant Isolation Layer                 │    │
  │  │  - Tenant ID management                         │    │
  │  │  - Data isolation guarantees                    │    │
  │  │  - Access control                               │    │
  │  └──────────────────┬──────────────────────────────┘    │
  │                      ▼                                   │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │           Lexicon Storage Engine                 │    │
  │  │  ┌────────────┐  ┌────────────┐  ┌──────────┐  │    │
  │  │  │   Local    │  │    Edge    │  │  Cloud   │  │    │
  │  │  │   SQLite   │  │  Workers   │  │   Store  │  │    │
  │  │  └──────┬─────┘  └──────┬─────┘  └────┬─────┘  │    │
  │  │         └────────────┬───┴──────────────┘       │    │
  │  └──────────────────────┼──────────────────────────┘    │
  │                      ▼                                   │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │           Learning & Adaptation Engine           │    │
  │  │  - Confidence progression                       │    │
  │  │  - Usage tracking                               │    │
  │  │  - Conflict resolution                          │    │
  │  │  - Decay management                             │    │
  │  └──────────────────┬──────────────────────────────┘    │
  │                      ▼                                   │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │           Sync & Replication                     │    │
  │  │  - Local ←→ Edge sync                           │    │
  │  │  - Conflict resolution                          │    │
  │  │  - Version control                              │    │
  │  └─────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────┘

  Database Schema

  -- Tenant memory schema
  CREATE TABLE tenant_lexicons (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      tenant_id VARCHAR(255) NOT NULL,
      original_text VARCHAR(255) NOT NULL,
      corrected_text VARCHAR(255) NOT NULL,
      confidence DECIMAL(3,2) DEFAULT 0.5,
      source VARCHAR(20) NOT NULL, -- 'user', 'retrieval', 'llm', 'context'
      source_details JSONB,
      usage_count INTEGER DEFAULT 0,
      last_used TIMESTAMP,
      created_at TIMESTAMP DEFAULT NOW(),
      updated_at TIMESTAMP DEFAULT NOW(),
      expires_at TIMESTAMP,
      context_pattern VARCHAR(500), -- For context-specific rules
      is_negative BOOLEAN DEFAULT FALSE, -- What NOT to correct

      UNIQUE KEY idx_tenant_original (tenant_id, original_text),
      INDEX idx_tenant_confidence (tenant_id, confidence),
      INDEX idx_expires (expires_at)
  );

  -- Usage tracking for learning
  CREATE TABLE lexicon_usage (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      lexicon_id UUID REFERENCES tenant_lexicons(id),
      job_id UUID NOT NULL,
      was_accepted BOOLEAN,
      was_modified BOOLEAN,
      modified_to VARCHAR(255),
      used_at TIMESTAMP DEFAULT NOW(),
      INDEX idx_lexicon (lexicon_id),
      INDEX idx_job (job_id)
  );

  -- Conflict resolution history
  CREATE TABLE lexicon_conflicts (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      tenant_id VARCHAR(255) NOT NULL,
      original_text VARCHAR(255) NOT NULL,
      candidate_1 VARCHAR(255),
      candidate_2 VARCHAR(255),
      resolution VARCHAR(255),
      resolution_method VARCHAR(20), -- 'user', 'confidence', 'recency'
      resolved_at TIMESTAMP DEFAULT NOW()
  );

  Implementation

  # app/services/tenant_memory.py

  from typing import Dict, Optional, List
  import sqlite3
  from datetime import datetime, timedelta
  from dataclasses import dataclass
  import json

  @dataclass
  class LexiconEntry:
      original: str
      corrected: str
      confidence: float
      source: str
      usage_count: int
      context_pattern: Optional[str] = None
      is_negative: bool = False

  class TenantMemory:
      def __init__(self, 
                   storage_backend: str = "sqlite",
                   sync_interval: int = 300):
          self.storage = self._init_storage(storage_backend)
          self.sync_interval = sync_interval
          self.confidence_increment = 0.1
          self.confidence_decay = 0.05
          self.max_confidence = 0.95
          self.min_confidence = 0.3

      def learn_correction(
          self,
          tenant_id: str,
          original: str,
          corrected: str,
          source: str,
          context: Optional[str] = None
      ) -> None:
          """Learn a new correction or update existing"""

          existing = self._get_entry(tenant_id, original)

          if existing:
              if existing.corrected == corrected:
                  # Reinforce existing correction
                  new_confidence = min(
                      existing.confidence + self.confidence_increment,
                      self.max_confidence
                  )
                  self._update_confidence(tenant_id, original, new_confidence)
              else:
                  # Conflict - need resolution
                  self._handle_conflict(
                      tenant_id, original,
                      existing.corrected, corrected,
                      source
                  )
          else:
              # New correction
              self._add_entry(
                  tenant_id, original, corrected,
                  source, initial_confidence=0.5
              )

      def get_corrections(
          self,
          tenant_id: str,
          threshold: float = 0.7
      ) -> Dict[str, str]:
          """Get high-confidence corrections for tenant"""

          query = """
              SELECT original_text, corrected_text, confidence
              FROM tenant_lexicons
              WHERE tenant_id = ? 
              AND confidence >= ?
              AND is_negative = FALSE
              AND (expires_at IS NULL OR expires_at > ?)
              ORDER BY confidence DESC, usage_count DESC
          """

          results = self.storage.execute(
              query,
              (tenant_id, threshold, datetime.now())
          )

          return {row[0]: row[1] for row in results}

      def track_usage(
          self,
          tenant_id: str,
          original: str,
          was_accepted: bool
      ) -> None:
          """Track whether correction was accepted"""

          if was_accepted:
              # Increase confidence
              self._adjust_confidence(tenant_id, original, self.confidence_increment)
          else:
              # Decrease confidence
              self._adjust_confidence(tenant_id, original, -self.confidence_decay)

      def _handle_conflict(
          self,
          tenant_id: str,
          original: str,
          existing_correction: str,
          new_correction: str,
          source: str
      ) -> None:
          """Resolve conflicting corrections"""

          if source == "user":
              # User correction takes precedence
              self._update_entry(tenant_id, original, new_correction, 0.8)
              # Mark old as negative example
              self._add_negative_example(tenant_id, original, existing_correction)
          else:
              # Keep higher confidence correction
              existing_conf = self._get_confidence(tenant_id, original)
              if existing_conf < 0.6:
                  self._update_entry(tenant_id, original, new_correction, 0.6)

  Edge Storage with Cloudflare Workers

  // edge-worker.js - Cloudflare Worker for global distribution

  export default {
    async fetch(request, env) {
      const { pathname } = new URL(request.url);

      if (pathname.startsWith('/lexicon/')) {
        const tenantId = pathname.split('/')[2];

        if (request.method === 'GET') {
          // Retrieve tenant lexicon from KV
          const lexicon = await env.LEXICONS.get(
            `tenant:${tenantId}`,
            { type: 'json' }
          );

          return new Response(JSON.stringify(lexicon || {}), {
            headers: { 'content-type': 'application/json' }
          });
        }

        if (request.method === 'PUT') {
          // Update tenant lexicon
          const updates = await request.json();

          // Get existing lexicon
          const existing = await env.LEXICONS.get(
            `tenant:${tenantId}`,
            { type: 'json' }
          ) || {};

          // Merge updates
          const merged = this.mergeLexicons(existing, updates);

          // Store back to KV
          await env.LEXICONS.put(
            `tenant:${tenantId}`,
            JSON.stringify(merged),
            { expirationTtl: 86400 * 30 } // 30 days
          );

          return new Response('OK');
        }
      }
    },

    mergeLexicons(existing, updates) {
      // Conflict resolution logic
      const merged = { ...existing };

      for (const [key, value] of Object.entries(updates)) {
        if (!merged[key] || value.confidence > merged[key].confidence) {
          merged[key] = value;
        }
      }

      return merged;
    }
  };

  Integration Points

  API Gateway Updates

  # app/api/routers/clean.py updates

  @router.post("/v1/clean/")
  async def clean_subtitles(
      request: CleanRequest,
      tenant_id: str = Header(None, alias="X-Tenant-ID")
  ) -> CleanResponse:
      """Enhanced cleaning with Layers 3-6"""

      # Layer 1-2: Existing hygiene and stabilization
      cleaned_doc = await cleaner.clean(document)

      # Layer 3: Context extraction
      if request.context_sources:
          context_lexicon = await context_extractor.extract_context(
              request.context_sources
          )
          cleaned_doc = apply_lexicon(cleaned_doc, context_lexicon)

      # Layer 4: On-demand retrieval
      if request.enable_retrieval:
          retrieval_results = await retriever.process_document(
              cleaned_doc.segments,
              region_hint=request.region
          )
          cleaned_doc = apply_retrievals(cleaned_doc, retrieval_results)

      # Layer 5: LLM selector for ambiguous cases
      if request.enable_llm and ambiguous_entities:
          selections = await llm_selector.select_entities(ambiguous_entities)
          cleaned_doc = apply_selections(cleaned_doc, selections)

      # Layer 6: Apply and learn from tenant memory
      if tenant_id:
          tenant_corrections = memory.get_corrections(tenant_id)
          cleaned_doc = apply_corrections(cleaned_doc, tenant_corrections)

          # Learn from this job
          for correction in applied_corrections:
              memory.learn_correction(
                  tenant_id,
                  correction.original,
                  correction.corrected,
                  correction.source
              )

      return CleanResponse(
          content=cleaned_doc,
          layers_applied=["hygiene", "stabilization", "context", "retrieval"],
          corrections_made=len(applied_corrections)
      )

  Performance & Monitoring

  Metrics Dashboard

  # app/infra/metrics.py

  LAYER_METRICS = {
      'layer3_context_extracted': Histogram('layer3_entities_extracted'),
      'layer3_extraction_time': Histogram('layer3_extraction_ms'),
      'layer4_spans_detected': Counter('layer4_suspicious_spans'),
      'layer4_retrieval_success': Counter('layer4_successful_retrievals'),
      'layer5_llm_calls': Counter('layer5_llm_invocations'),
      'layer5_selection_confidence': Histogram('layer5_confidence_scores'),
      'layer6_lexicon_size': Gauge('layer6_tenant_lexicon_entries'),
      'layer6_hit_rate': Histogram('layer6_lexicon_hit_rate')
  }

  Deployment Strategy

  Phase 1: Layer 3 Rollout (Week 1)

  - Deploy context extraction with existing infrastructure
  - Test with beta users providing show notes/URLs
  - Monitor extraction quality and performance

  Phase 2: Layer 4 Limited Beta (Week 2-3)

  - Enable retrieval for 10% of requests
  - Focus on high-confidence corrections only
  - Cache aggressively, monitor API costs

  Phase 3: Layer 5 Premium Feature (Week 4)

  - Deploy local LLM for privacy-conscious users
  - Offer OpenAI integration for premium tier
  - A/B test correction quality

  Phase 4: Layer 6 Production (Week 5-6)

  - Enable tenant memory for all users
  - Deploy edge workers for global distribution
  - Monitor learning effectiveness