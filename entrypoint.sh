#!/bin/bash
set -e

# Start Ollama in the background
ollama serve &

# Wait until Ollama is ready
echo "Waiting for Ollama..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
  sleep 1
done

# Pull models if not already present
ollama pull nomic-embed-text
ollama pull gemma4:e2b

echo "Models ready. Running Python script..."

streamlit run app.py