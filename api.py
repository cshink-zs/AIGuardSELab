import uuid
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent_core import build_agent, chat as agent_chat

load_dotenv()

app = FastAPI(title="AI Guard Chat API", version="1.0.0")

# Agents are expensive to build; cache by (provider, model, mode)
_agent_cache: dict = {}


async def _get_agent(provider: str, model: str, mode: str):
    key = f"{provider}:{model}:{mode}"
    if key not in _agent_cache:
        _agent_cache[key] = await build_agent(provider, model, mode)
    return _agent_cache[key]


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    provider: str = "Anthropic"
    model: str = "claude-haiku-4-5-20251001"
    mode: str = "DAS"          # "DAS" (AI Guard middleware) or "Proxy" (Zscaler reverse proxy)
    inspect_prompt: bool = True
    inspect_response: bool = True


class ChatResponse(BaseModel):
    response: str
    thread_id: str


@app.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    try:
        agent, _ = await _get_agent(req.provider, req.model, req.mode)
        response, _ = await agent_chat(
            agent,
            req.message,
            thread_id,
            inspect_prompt=req.inspect_prompt,
            inspect_response=req.inspect_response,
        )
        return ChatResponse(response=response, thread_id=thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
