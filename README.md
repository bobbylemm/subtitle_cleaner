# Universal Subtitle Corrector 🌍

A high-accuracy, domain-agnostic subtitle correction system powered by LLMs and a self-improving Knowledge Base.

## 🚀 Features

*   **Context-Aware Correction**: Uses a two-stage LLM pipeline to understand the full context (Topic, Industry, Country) before correcting.
*   **Knowledge Base (Memory)**: Learns from your corrections. If you fix "Mecano" -> "Upamecano" once, it remembers it forever.
*   **Multi-Dimensional Context**: Distinguishes between terms based on context (e.g., "Chip" in Tech vs. Sports).
*   **Style Preservation**: Intelligently preserves slang, nicknames, and speaker intent.
*   **Global Consistency**: Ensures names like "Man United" or "Frenkie de Jong" are consistent throughout the file.

## 🛠️ Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Set Environment Variables**:
    ```bash
    export OPENAI_API_KEY="your-key"
    export OPENAI_MODEL="gpt-4o" # Recommended for best results
    ```

## 📖 Usage

### 1. Run Correction (CLI)
Correct a subtitle file automatically.

```bash
python scripts/run_correction.py input.srt output.srt
```

### 2. Teach the System (Learning)
Feed approved corrections back into the Knowledge Base to improve future accuracy.

```bash
python scripts/learn_corrections.py original.srt corrected.srt --topic Football --industry Sports --country Germany
```
*   **--topic**: Broad category (e.g., Football, Politics).
*   **--industry**: Sector (e.g., Sports, Tech).
*   **--country**: Local context (e.g., Germany, Nigeria).

### 3. API Server
Run the FastAPI server for production use.

```bash
uvicorn app.main:app --reload
```
*   **Endpoint**: `POST /api/v1/universal/universal-correct`
*   **Params**: `file`, `topic`, `industry`, `country`

## 🧠 Knowledge Base

The system uses a SQLite database (`knowledge_base.db`) to store corrections.
It uses a **Weighted Retrieval Algorithm** to find the best correction for the current context:
*   **Country Match**: +10 points (Highest priority for local entities).
*   **Topic/Industry Match**: +5 points.

## 🏗️ Architecture

1.  **Context Analysis (Stage 1)**: The LLM reads the file to extract a `ContextManifest` (Topic, Genre, Entities, Style Guide).
2.  **Knowledge Retrieval**: The system fetches relevant corrections from the DB based on the manifest.
3.  **Chunk Correction (Stage 2)**: The file is processed in chunks, injecting the Manifest + KB Corrections into the prompt.
4.  **Global Consistency**: A final pass ensures all entities are spelled consistently.
