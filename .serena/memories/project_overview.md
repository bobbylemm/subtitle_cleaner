# Project Overview

## Purpose
The Subtitle Cleaner is a FastAPI-based application for cleaning and perfecting SRT/WebVTT subtitle files across multiple languages. It provides intelligent subtitle processing with ML-enhanced cleaning capabilities to achieve 98% accuracy target.

## Core Functionality
- **Subtitle Format Support**: SRT and WebVTT formats
- **Multi-language Processing**: English, Spanish, French, German, Italian, Portuguese, Dutch
- **Intelligent Cleaning**: Rule-based + ML enhancement pipeline
- **Context-Aware Processing**: Supports external context sources for better entity recognition
- **Tenant Memory**: Per-tenant learning and customization
- **Quality Assurance**: Comprehensive validation and quality scoring

## Architecture Layers
1. **Layer 1-2**: Basic rule-based cleaning (parsing, validation, normalization)
2. **Layer 3**: Context extraction from user-provided sources (URLs, files, text)
3. **Layer 4**: On-demand retrieval from Wikipedia/Wikidata for suspicious entities
4. **Layer 5**: LLM selection for ambiguous cases (optional, requires OpenAI API key)
5. **Layer 6**: Tenant memory and learning system with PostgreSQL storage

## Target Accuracy
- Rule-based baseline: 90% accuracy
- ML-enhanced target: 98% accuracy
- Deterministic processing with strict guardrails
- Full audit trail of changes