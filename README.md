---
title: HH Goa 2026 Voice RAG
sdk: docker
app_port: 8000
python_version: "3.11"
---
# HH Goa 2026: Voice-Enabled RAG System

A multilingual, voice-first RAG pipeline designed for deployment on Hugging Face Spaces with a decoupled static frontend on GitHub Pages.

## Architecture

```text
                 GitHub Pages
                Static Frontend
                     │
                     │ HTTPS
                     ▼
              Hugging Face Space
                 Python Backend
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      Sarvam       RAG          Gemini
        │            │            │
        └────────────┼────────────┘
                     ▼
                 JSON result
                     │
                     ▼
                Web Frontend
```

## Deployment

This repository is split into two halves:

1. **Frontend (`/frontend`)**: A pure HTML/CSS/JS single-page application.
   - Deployed automatically via GitHub Actions to GitHub Pages.
   - Requires no Python or server-side rendering.
   - Configurable via `frontend/js/config.js` to point to the backend URL.

2. **Backend (`app.py` & `/src`)**: A FastAPI server.
   - Designed to run as a Docker Space on Hugging Face.
   - Exposes REST endpoints (`/api/v1/voice`, `/api/v1/text`, `/health`).
   - Connects to Qdrant (local), Gemini API, and Sarvam STT.

## Local Development

```bash
# 1. Start the backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# 2. Start the frontend
cd frontend
python -m http.server 3000
```

Open `http://localhost:3000` in your browser.

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys for local development.
**Never commit `.env` to Git.**

Required secrets on Hugging Face Spaces:
- `GEMINI_API_KEY`: Google Gemini token.
- `SARVAM_API_KEY`: Sarvam AI token.
- `FRONTEND_ORIGIN`: Your GitHub Pages URL (e.g., `https://username.github.io`) to restrict CORS.

## Performance & Latency

STT latency has been heavily optimized:
- **Streaming WebSockets** (where supported): `< 200ms` wait from speech end.
- **REST Fallback**: ~`872ms` P50 for 1-second chunks.
- **Generation**: Constrained by Gemini 3.6 Flash free tier (~`4.9s`).

## Limitations

- Heavy embeddings or huge datasets are not committed. Ensure you run the ingestion pipeline (`tests/test_harness.py` or similar scripts) to populate the local Qdrant database `data/qdrant_db` before deploying, or use a managed Qdrant cloud instance.
- The Gemini API free tier restricts traffic to 20 requests per day.
