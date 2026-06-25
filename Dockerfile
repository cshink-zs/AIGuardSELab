FROM python:3.14.2

# Install dependencies needed for Ollama
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
# Download and extract Ollama binary (now a .tar.zst archive)
# Download first, decompress with zstd, then extract with tar
ARG TARGETARCH
RUN curl -fsSL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-${TARGETARCH}.tar.zst \
    -o /tmp/ollama.tar.zst && \
    zstd -d /tmp/ollama.tar.zst -o /tmp/ollama.tar && \
    tar -xf /tmp/ollama.tar -C /usr && \
    rm /tmp/ollama.tar.zst /tmp/ollama.tar


# Set working directory
WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy Python files
COPY app.py .
COPY aiguard_utils.py .

# Copy Streamlit config
COPY ./.streamlit/config.toml ./.streamlit/config.toml

EXPOSE 8501

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]