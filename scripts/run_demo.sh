#!/bin/bash
export APP_MODE=demo
export PYTHONPATH=.
echo "Starting HH Goa Voice RAG Video Demo..."
.venv/bin/python gradio_app.py
