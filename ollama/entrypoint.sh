#!/bin/bash

# Start Ollama server in the background
ollama serve &
OLLAMA_PID=$!

# Wait until Ollama is ready to accept requests
echo "⏳ Waiting for Ollama to be ready..."
until curl -s http://localhost:11434 > /dev/null 2>&1; do
  sleep 1
done
echo "✅ Ollama is up!"

# Pull models (skips if already cached in the volume)
echo "📥 Pulling gemma4:e2b..."
ollama pull gemma4:e2b

echo "📥 Pulling nomic-embed-text..."
ollama pull nomic-embed-text

echo "🚀 All models loaded. Ollama is ready!"

# Hand control back to the Ollama server process
wait $OLLAMA_PID
