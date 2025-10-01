# Phase 2: Contextual Understanding Layer - Design Document

## Executive Summary
Phase 2 implements semantic understanding across subtitle segments, moving beyond isolated segment processing to understand narrative structure, speaker relationships, and discourse coherence. Based on extensive research of 2024 state-of-the-art techniques, this design prioritizes lightweight, deterministic, and fast inference using ONNX-optimized models.

## Architecture Overview

### Core Principles
1. **Lightweight First**: Use smallest effective models (all-MiniLM-L6-v2 with 384 dims vs BERT-large with 1024)
2. **ONNX Optimization**: 7.5-10x speedup on CPU inference
3. **Incremental Processing**: Process segments in sliding windows to maintain <5s for 60min constraint
4. **Explainable**: Every decision tracked with confidence scores
5. **Fallback Strategies**: Rule-based alternatives when ML models unavailable

## Component Design

### 1. Sliding Window Context Manager
**Purpose**: Maintain contextual awareness across segments without loading entire document in memory

**Implementation**:
```python
class SlidingWindowContext:
    def __init__(self, window_size=10, overlap=3):
        self.window_size = window_size  # segments
        self.overlap = overlap
        self.embedding_cache = LRUCache(maxsize=1000)
```

**Key Features**:
- LRU cache for embeddings (avoid recomputation)
- Configurable window size based on available memory
- Overlap ensures smooth transitions
- Processing time: ~0.1ms per segment lookup

### 2. Lightweight Speaker Tracking
**Purpose**: Track speaker changes without heavy diarization models

**Approach**: 
- Use sentence embeddings to detect style changes (different speakers have different linguistic patterns)
- Cluster similar speaking styles
- No audio processing required (works on text only)

**Implementation**:
```python
class LightweightSpeakerTracker:
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model)
        self.speaker_embeddings = []
        self.similarity_threshold = 0.85
```

**Performance**: 
- 384-dim embeddings (vs 768/1024 for larger models)
- ~2ms per segment
- Memory: ~150KB for 1000 segments

### 3. Topic Segmentation with Coherence Scoring
**Purpose**: Group related segments into coherent narrative sections

**Approach**: Hybrid TextTiling + Sentence Transformers
- Compute semantic similarity between adjacent segments
- Detect topic boundaries when similarity drops below threshold
- Use moving average to smooth transitions

**Implementation**:
```python
class TopicSegmenter:
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.boundary_threshold = 0.6
        self.smoothing_window = 3
```

**Key Metrics**:
- Cosine similarity (normalized embeddings → dot product)
- Topic coherence score: average similarity within topic
- Processing: ~3ms per segment

### 4. Efficient Coreference Resolution
**Purpose**: Resolve pronouns and references across segments

**Approach**: Lightweight rule-based + embedding similarity
- Track entity mentions in sliding window
- Simple pronoun resolution rules
- Fallback to embedding similarity for ambiguous cases

**Implementation**:
```python
class EfficientCoreferenceResolver:
    def __init__(self):
        self.pronoun_rules = {
            'he': 'male_person',
            'she': 'female_person',
            'it': 'object_or_concept',
            'they': 'plural_or_neutral'
        }
        self.entity_buffer = deque(maxlen=20)
```

**Performance**:
- Rule-based: <0.1ms per pronoun
- Embedding fallback: ~1ms per ambiguous case
- Memory: ~10KB for entity buffer

### 5. ONNX-Optimized Semantic Scorer
**Purpose**: Score semantic coherence between segments

**Implementation**:
```python
class ONNXSemanticScorer:
    def __init__(self, model_path="models/minilm-onnx"):
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L6")
```

**Optimizations**:
- ONNX quantization (int8): 4x model size reduction
- Batch processing: Process multiple segments together
- Graph optimizations: Node fusion, layer normalization
- Expected speedup: 7-10x over PyTorch

## Integration Architecture

```
Input Segments
     ↓
[Sliding Window Manager]
     ↓
[Parallel Processing]
    ├── Speaker Tracking
    ├── Topic Segmentation
    ├── Coreference Resolution
    └── Coherence Scoring
     ↓
[Aggregation Layer]
     ↓
[Context-Enhanced Segments]
```

## Performance Budget

For 60-minute video (~3600 segments):
- Embedding generation: 3600 × 2ms = 7.2s
- Topic segmentation: 3600 × 1ms = 3.6s  
- Speaker tracking: 3600 × 0.5ms = 1.8s
- Coreference: 3600 × 0.2ms = 0.72s
- **Total: ~13.3s** (can be optimized to <5s with batching and caching)

With optimizations:
- Batch processing (32 segments): 3.5x speedup
- ONNX optimization: 7x speedup
- Caching (30% cache hits): 1.4x speedup
- **Optimized total: ~2.7s**

## Memory Requirements
- Embeddings cache: 10MB (1000 segments × 384 dims × 4 bytes)
- Models: ~50MB (MiniLM ONNX quantized)
- Working memory: 20MB
- **Total: ~80MB RAM**

## Implementation Priority

1. **Week 1**: Core infrastructure
   - Sliding window context manager
   - ONNX model setup and optimization
   - Basic embedding generation

2. **Week 2**: Context features
   - Topic segmentation
   - Speaker tracking  
   - Coherence scoring

3. **Week 3**: Advanced features
   - Coreference resolution
   - Cross-segment entity linking
   - Integration and optimization

## Fallback Strategies

When models unavailable:
1. **Speaker Tracking**: Punctuation and style heuristics
2. **Topic Segmentation**: Keyword overlap and sentence length variance
3. **Coreference**: Simple pronoun rules and recency heuristics
4. **Coherence**: Word overlap and TF-IDF similarity

## Success Metrics

- Processing time: <5s for 60-minute content
- Memory usage: <100MB
- Coherence improvement: 20% reduction in pronoun ambiguity
- Topic boundary accuracy: >80% precision
- Speaker consistency: >85% accuracy in maintaining speaker identity

## Risk Mitigation

1. **Performance degradation**: Pre-compute embeddings during idle time
2. **Memory constraints**: Use streaming processing with fixed buffers
3. **Model availability**: Ship quantized ONNX models with application
4. **Accuracy issues**: Adjustable confidence thresholds with user overrides

## Next Steps

1. Implement SlidingWindowContext class
2. Convert MiniLM model to ONNX format
3. Build topic segmentation with TextTiling approach
4. Create lightweight speaker tracker
5. Integrate with existing Phase 1 pipeline