# Hacker House Goa 2026: Voice-Enabled RAG

A low-latency, voice-enabled Retrieval-Augmented Generation (RAG) system built from scratch to query the `ai4bharat/MSMARCO-XI` dataset.

## Architecture

1. **Ingestion & Data Preprocessing**:
    - Script: `src/data/ingest.py`
    - Logic: Streams dataset dynamically to avoid OOM crashes (an issue identified in earlier prototypes), safely parsing the nested translation format and producing deduplicated chunkable files.
2. **Chunking**:
    - Location: `src/chunking/`
    - Logic: Uses a `ChunkingRouter` that evaluates text through multiple simultaneous strategies (fixed-overlap + sentence boundaries) to preserve maximum semantic context.
3. **Retrieval**:
    - Embeddings: `intfloat/multilingual-e5-small` with query/passage prefixing.
    - Vector Store: `FAISS` using the `IndexHNSWFlat` algorithm. Extremely fast in-memory search achieving <5ms indexing searches.
    - Re-ranking: `ms-marco-MiniLM-L-6-v2` cross-encoder for final sorting.
4. **Guardrails**:
    - **Pre-Retrieval**: Regex/Heuristic blocking of unsafe/jailbreak terms.
    - **Post-Generation**: Enforces strict LLM formatting ("I don't have enough information") and citation checks.
5. **Orchestration**:
    - A strict pipeline (`src/pipeline/harness.py`) enforcing data contracts and calculating P-latency for each component stage.

## Why Sarvam over ElevenLabs?

We explicitly chose **Sarvam AI (Saaras v3)** for the Speech-to-Text component. While ElevenLabs excels at generation (TTS) and some global ASR, this project specifically demands robustness for 14 Indic Languages. Sarvam's models are natively trained on Indic data distributions and achieve noticeably lower WER (Word Error Rates) on Indian accents and code-switched (Hinglish/Tanglish) audio.

## Latency Rule & Benchmarks

The project requirements specify a `<200ms` target for the pipeline.
**Engineering Transparency Note**: While the internal RAG architecture (FAISS HNSW Retrieval + Embedding + Reranking) easily operates in `<100ms`, relying on a cloud LLM (like Gemini 3.5 Flash or Groq) for the generation phase inherently introduces `~500ms+` of network overhead when waiting for the full string to generate.

To run the benchmarking suite locally:
```bash
python src/benchmark/run_benchmarks.py
```

## Running the API

We use a FastAPI server capable of accepting text or `.wav` inputs.

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```
