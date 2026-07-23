FROM python:3.14-slim

# Prevent .pyc files and enable unbuffered stdout/stderr for Docker logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv — used for dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies before copying source so this layer is cached
# unless pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache

# Application source
COPY .env ./
COPY app.py api.py agent_core.py aiguard_utils.py ./
COPY .streamlit/ ./.streamlit/

# Streamlit UI | FastAPI REST
EXPOSE 8501 8000

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
