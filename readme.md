# AI Guard Demo Agent

A demo project showing how [Zscaler AI Guard](https://api.zseclipse.net) can be integrated into a LangChain / LangGraph agent pipeline. AI Guard inspects prompts (IN) and responses (OUT) and can **allow**, **mask**, or **block** traffic before it reaches — or leaves — the LLM.

It ships two entry points that share the same agent core:

| Component | File | Port | Purpose |
|-----------|------|------|---------|
| Streamlit UI | `app.py` | `8501` | Interactive chat demo with sidebar controls (mode, provider, MCP tools, RAG upload) |
| FastAPI REST | `api.py` | `8000` | `POST /v1/chat` endpoint + `GET /health` |

Both are built on `agent_core.py` (agent construction + chat loop) and `aiguard_utils.py` (the AI Guard HTTP client).

## Inspection modes

- **DAS** (Detection as a Service) — AI Guard runs as LangChain middleware. The prompt/response is sent to the AI Guard policy engine and blocked/masked according to the policy. Requires `GUARDRAIL_DAS_API_KEY` **and** `GUARDRAIL_DAS_POLICY_ID`.
- **Proxy** — traffic is routed through the Zscaler reverse proxy (`proxy.zseclipse.net`) via a request header. Requires `GUARDRAIL_PROXY_API_KEY`.

A mode only appears in the UI if its required environment variables are set.

---

## Prerequisites

- **Python 3.14+** (see `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** — used for dependency management and running the app
- **[Ollama](https://ollama.com/)** running locally — **required even when using Anthropic**, because the RAG vector store uses the `nomic-embed-text` embedding model. Pull it once:
  ```bash
  ollama pull nomic-embed-text
  ```
  (If you plan to use the local `Ollama` LLM provider too, also pull that model.)
- API keys — at minimum an `ANTHROPIC_API_KEY`, plus the AI Guard key(s) for whichever inspection mode you want to demo.

---

## Configure environment variables

The app loads configuration from a `.env` file in the project root (via `python-dotenv`). Create one by copying the example below.

> ⚠️ `.env` contains secrets — it should **never** be committed. Confirm it is in `.gitignore`.

### `.env` example

```dotenv
# ── LLM provider ─────────────────────────────────────────────
# Required when using Anthropic as the provider (the default).
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx

# Optional — only if you enable the OpenAI provider.
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# ── AI Guard: DAS mode (Detection as a Service) ──────────────
# Both are required for "DAS" mode to appear and work.
GUARDRAIL_DAS_API_KEY=your-das-bearer-token
GUARDRAIL_DAS_POLICY_ID=your-policy-id

# ── AI Guard: Proxy mode ─────────────────────────────────────
# Required for "Proxy" mode.
GUARDRAIL_PROXY_API_KEY=your-proxy-api-key

# ── LangSmith tracing (all optional) ─────────────────────────
# LANGSMITH_TRACING=true
# LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# LANGSMITH_API_KEY=lsv2_xxxxxxxxxxxxxxxxxxxxxxxx
# LANGSMITH_PROJECT=ai-guard-demo
```

### Variable reference

| Variable | Required? | Used for |
|----------|-----------|----------|
| `ANTHROPIC_API_KEY` | Yes (for Anthropic provider) | Authenticates calls to Claude models |
| `GUARDRAIL_DAS_API_KEY` | For DAS mode | Bearer token for the AI Guard policy engine |
| `GUARDRAIL_DAS_POLICY_ID` | For DAS mode | Which AI Guard policy to enforce |
| `GUARDRAIL_PROXY_API_KEY` | For Proxy mode | `X-ApiKey` header sent to the Zscaler proxy |
| `OPENAI_API_KEY` | Optional | Only if using the OpenAI provider |
| `LANGSMITH_TRACING` / `LANGSMITH_ENDPOINT` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | Optional | LangSmith run tracing/observability |
| `OLLAMA_HOST` | Optional | Overrides the Ollama base URL (set automatically in Docker to reach the host) |

> 💡 The Streamlit sidebar has an **AI Guard Configuration** section that lets you edit any `GUARDRAIL_*` value at runtime and save it back to `.env`.

---

## Run locally

Install dependencies (creates a `.venv` from `uv.lock`):

```bash
uv sync
```

Make sure Ollama is running (`ollama serve`) and `nomic-embed-text` is pulled.

### Streamlit UI

```bash
uv run streamlit run app.py
```

Then open http://localhost:8501.

### FastAPI REST API

```bash
uv run uvicorn api:app --reload --port 8000
```

- Health check: `GET http://localhost:8000/health`
- Interactive docs: http://localhost:8000/docs

Example request:

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!",
    "provider": "Anthropic",
    "model": "claude-haiku-4-5-20251001",
    "mode": "DAS",
    "inspect_prompt": true,
    "inspect_response": true
  }'
```

---

## Run with Docker

The `Dockerfile` + `docker-compose.yml` run **both** the Streamlit UI and the FastAPI server in a single container. The `.env` file is loaded via `env_file`, and `OLLAMA_HOST` is set to `http://host.docker.internal:11434` so the container can reach Ollama running on your host.

```bash
docker compose up --build
```

- Streamlit UI → http://localhost:8501
- FastAPI REST → http://localhost:8000

> On **Linux**, `host.docker.internal` isn't available by default — add `extra_hosts: ["host.docker.internal:host-gateway"]` to the `app` service (or run Ollama in its own container; a commented-out service is included in `docker-compose.yml`).

---

## Project layout

```
app.py            # Streamlit UI
api.py            # FastAPI REST layer
agent_core.py     # Agent construction, AI Guard middleware, chat loop
aiguard_utils.py  # AIGuardClient — HTTP client for the AI Guard policy API
Dockerfile        # Builds the combined UI + API image
docker-compose.yml
entrypoint.sh     # Starts uvicorn (bg) + streamlit (fg) in the container
pyproject.toml    # Dependencies (managed by uv)
.streamlit/       # Streamlit theme config
```
