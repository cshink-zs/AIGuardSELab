#!/bin/bash
set -e

echo "Starting FastAPI server on port 8000..."
uv run uvicorn api:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit on port 8501..."
exec uv run streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true
