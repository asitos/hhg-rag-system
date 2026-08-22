#!/bin/bash
export APP_MODE=demo
export PYTHONPATH=.

echo "=========================================="
echo " Starting HH Goa Voice RAG Video Demo"
echo "=========================================="
echo "Mode: DEMO (Backend RAG + Mocks)"
echo ""
echo "Please open your browser to:"
echo "http://localhost:8000"
echo "=========================================="
.venv/bin/python app.py
