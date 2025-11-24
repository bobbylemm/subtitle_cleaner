# Universal Subtitle Corrector 🌍

A high-accuracy, domain-agnostic subtitle correction system powered by LLMs and a self-improving Knowledge Base.

## 🚀 Quick Start

### Prerequisites
*   Python 3.13+
*   OpenAI API Key

### Installation

1.  **Clone & Install**
    ```bash
    git clone <repo>
    cd clean_srt
    pip install -r requirements.txt
    ```

2.  **Configure**
    Copy `.env.sample` to `.env` and set your `OPENAI_API_KEY`.
    ```bash
    cp .env.sample .env
    ```

3.  **Run**
    ```bash
    uvicorn app.main:app --reload
    ```
    Open [http://localhost:8000](http://localhost:8000) to use the Web UI.

## 🛠️ Features

*   **Universal Correction**: Uses LLM (GPT-4o) to fix typos, grammar, and phonetic errors while preserving slang.
*   **Context-Aware**: Automatically detects Topic, Industry, and Country.
*   **Knowledge Base**: Learns from corrections (stored in `knowledge_base.db`).
*   **Web UI**: Simple Drag & Drop interface.
*   **API**: REST API for integration.

## 📚 API Documentation

Once running, visit:
*   Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
*   ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

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
