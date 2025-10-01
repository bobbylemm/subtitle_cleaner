# ML-Enhanced Subtitle Processing Architecture

## Overview
Hybrid deterministic + constrained ML approach for 98% accuracy in subtitle cleaning.

## Architecture Layers

### 1. Core Pipeline (Deterministic)
```
Parse → Validate → Merge → Wrap → Normalize → Serialize
```

### 2. ML Enhancement Layer (Constrained)
```
┌─────────────────────────────────────────────┐
│  Input Segment                              │
├─────────────────────────────────────────────┤
│  1. Entity Protection (NER/Glossary Lock)   │
│  2. Punctuation & Truecasing Model          │
│  3. Grammar Error Correction Model          │
│  4. Context-Aware Filler Detection          │
│  5. Perplexity Scoring                      │
├─────────────────────────────────────────────┤
│  Gatekeeper Validation                      │
│  - Diff budget (<15% tokens changed)        │
│  - Edit type whitelist                      │
│  - Perplexity improvement check             │
│  - Entity preservation check                │
├─────────────────────────────────────────────┤
│  Output: Enhanced or Original (fallback)    │
└─────────────────────────────────────────────┘
```

## Model Components

### Punctuation & Truecasing
- **Model**: ByT5-small or NeMo P&C
- **Scope**: Punctuation, capitalization, diacritics
- **Size**: ~100MB quantized
- **Latency**: <50ms per segment

### Grammar Error Correction
- **Model**: GECToR (EN), mT5-small (multilingual)
- **Scope**: Articles, agreement, morphology
- **Size**: ~200MB quantized
- **Latency**: <100ms per segment

### Context Filler Detection
- **Model**: BiLSTM-CRF or DistilBERT
- **Scope**: Disambiguate fillers vs content
- **Size**: ~50MB
- **Latency**: <30ms per segment

### Perplexity Scorer
- **Model**: KenLM 5-gram or TinyBERT
- **Scope**: Quality assessment
- **Size**: ~100MB per language
- **Latency**: <20ms per segment

## Determinism Guarantees

1. **Fixed Seeds**: All models use fixed random seeds
2. **Temperature 0**: No sampling, always greedy decoding
3. **Versioned Models**: Pinned model versions
4. **Fallback Logic**: Original text if constraints violated
5. **Audit Trail**: Every ML edit logged with reason

## Performance Targets

- **Accuracy**: 98% (vs 90% rules-only)
- **Latency**: <500ms total per segment
- **Memory**: <2GB total with all models loaded
- **CPU-only**: Quantized models run on CPU
- **GPU-optional**: 10x faster with small GPU

## Language Support Priority

1. **Tier 1** (Full ML): EN, ES, FR
2. **Tier 2** (Partial ML): DE, IT, PT
3. **Tier 3** (Rules + Basic ML): NL, others