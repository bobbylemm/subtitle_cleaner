# Processing Capabilities

## Core Subtitle Processing

### Format Support
- **Input Formats**: SRT, WebVTT
- **Output Formats**: SRT, WebVTT
- **File Size Limits**: 10MB default (configurable)
- **Segment Limits**: 10,000 segments default (configurable)

### Language Support Tiers
**Tier 1 (Full ML Support)**:
- English, Spanish, French: All ML models available

**Tier 2 (Partial ML)**:
- German, Italian, Portuguese: Punctuation and grammar only

**Tier 3 (Basic ML)**:
- Dutch: Punctuation only

### Rule-Based Processing (Layers 1-2)
- **Parsing & Validation**: Robust SRT/WebVTT parsing with validation
- **Segment Merging**: Smart, aggressive, conservative, or disabled modes
- **Text Normalization**: Punctuation, casing, common error fixes
- **Filler Removal**: Language-specific filler word detection and removal
- **Line Wrapping**: Intelligent wrapping with punctuation and phrase preservation
- **Timing Optimization**: CPS (characters per second) optimization
- **Glossary Enforcement**: Custom term standardization

## Enhanced ML Processing (Layers 3-6)

### Layer 3: Context Extraction
- **Source Types**: URLs, files, raw text
- **Entity Recognition**: NER pipeline with confidence scoring
- **Authority Weighting**: Source reliability assessment
- **Conflict Resolution**: Multi-source entity disambiguation
- **Auto-Context Generation**: Automatic entity extraction from subtitle content

### Layer 4: On-Demand Retrieval
- **Knowledge Sources**: Wikipedia, Wikidata
- **Suspicious Entity Detection**: Automatic identification of entities needing verification
- **Regional Hints**: Location-aware entity resolution
- **Consensus Building**: Multi-source verification

### Layer 5: LLM Selection (Optional)
- **OpenAI Integration**: GPT-3.5-turbo for ambiguous cases
- **Confidence Thresholds**: Configurable minimum confidence levels
- **Fallback Strategy**: Rule-based fallback when LLM unavailable
- **Cost Optimization**: Selective use for complex disambiguation only

### Layer 6: Tenant Memory
- **PostgreSQL Storage**: Persistent tenant-specific corrections
- **Learning System**: Automatic improvement from corrections
- **Confidence Levels**: HIGH, MEDIUM, LOW confidence classifications
- **Cross-Session Learning**: Tenant preferences persist across API calls

## Quality Assurance

### Validation & Constraints
- **Edit Ratio Limits**: Maximum 15% token changes
- **Character Change Limits**: Maximum 8 character net change per segment
- **Entity Protection**: Preserve proper nouns, numbers, URLs
- **Perplexity Scoring**: Quality assessment with improvement requirements
- **Confidence Thresholds**: Minimum confidence for ML acceptance

### Performance Targets
- **Latency**: <500ms per segment (CPU), <50ms (GPU)
- **Memory Usage**: <2GB with all models loaded
- **Accuracy**: 98% vs human gold standard (target)
- **Determinism**: 100% reproducible outputs

## Limitations

### Current Constraints
- **Maximum File Size**: 10MB (configurable)
- **Maximum Segments**: 10,000 (configurable)
- **Processing Timeout**: 60 seconds default
- **Language Coverage**: Limited to 7 languages
- **Model Dependencies**: Requires significant memory for ML models

### API Dependencies
- **OpenAI API**: Required for Layer 5 LLM selection
- **Internet Access**: Required for Layer 4 retrieval
- **Database**: PostgreSQL required for tenant memory
- **Redis**: Required for caching and rate limiting