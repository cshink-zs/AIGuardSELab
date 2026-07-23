import asyncio
import contextvars
import os
from typing import Any

import requests
from aiguard_utils import AIGuardClient
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, after_agent, before_agent
from langchain.messages import AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

load_dotenv()

# Per-request inspection flags — set by chat() before invoking the agent
inspect_prompt_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar("inspect_prompt", default=True)
inspect_response_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar("inspect_response", default=True)

def _make_guard() -> AIGuardClient:
    return AIGuardClient(
        bearer_token=os.getenv("GUARDRAIL_DAS_API_KEY"),
        policy_id=os.getenv("GUARDRAIL_DAS_POLICY_ID"),
    )


guard = _make_guard()


def reconfigure_guard() -> None:
    """Recreate the guard client from the current environment variables."""
    global guard
    guard = _make_guard()


def ollama_available() -> bool:
    """Return True if a local Ollama server is reachable.

    RAG embeddings (and the optional Ollama LLM provider) depend on Ollama.
    We probe the /api/tags endpoint with a short timeout so the app degrades
    gracefully when Ollama isn't running.
    """
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=2)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def available_modes() -> list[str]:
    """Return inspection modes that have the required env vars configured."""
    modes = []
    if os.getenv("GUARDRAIL_DAS_API_KEY") and os.getenv("GUARDRAIL_DAS_POLICY_ID"):
        modes.append("DAS")
    if os.getenv("GUARDRAIL_PROXY_API_KEY"):
        modes.append("Proxy")
    return modes


@before_agent(can_jump_to=["end"])
def InspectPrompt(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if not inspect_prompt_ctx.get():
        return None
    if not state["messages"]:
        return None
    last = state["messages"][-1]
    if last.type != "human":
        return None
    try:
        guard.enforce(direction="IN", content=last.content)
    except ValueError as e:
        # Guard explicitly blocked the prompt
        return {"messages": [AIMessage(str(e))], "jump_to": "end"}
    except Exception as e:
        # Network error or unexpected failure — block and surface the error
        return {"messages": [AIMessage(f"AI Guard unavailable: {e}")], "jump_to": "end"}
    return None


@after_agent(can_jump_to=["end"])
def InspectResponse(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if not inspect_response_ctx.get():
        return None
    if not state["messages"]:
        return None
    last = state["messages"][-1]
    if last.type != "ai":
        return None
    try:
        guard.enforce(direction="OUT", content=last.content)
    except ValueError as e:
        # Guard explicitly blocked the response — replace via state update, not mutation
        return {"messages": [AIMessage(str(e))]}
    except Exception as e:
        return {"messages": [AIMessage(f"AI Guard unavailable: {e}")]}
    return None


DEFAULT_MCP_CONFIG: dict[str, dict] = {
    "dlptest": {
        "url": "https://mcp.dlptest.com/api/mcp/",
        "headers": {},
    },
}


async def _init_mcp_tools(mcp_config: dict[str, dict] | None = None) -> list:
    config = mcp_config if mcp_config is not None else DEFAULT_MCP_CONFIG
    if not config:
        return []
    client_config = {}
    for name, settings in config.items():
        entry: dict = {"transport": "http", "url": settings["url"]}
        if settings.get("headers"):
            entry["headers"] = settings["headers"]
        client_config[name] = entry
    client = MultiServerMCPClient(client_config)
    return await client.get_tools()


async def build_agent(provider: str, model: str, mode: str, mcp_config: dict[str, str] | None = None):
    """Create an agent and vector store for the given provider/model/mode.

    RAG requires Ollama (for the nomic-embed-text embeddings). When Ollama is
    unavailable, the agent is built without the knowledge-base tool and the
    returned vector store is None.
    """
    rag_enabled = ollama_available()

    vector_store = None
    rag_tools: list = []
    if rag_enabled:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = InMemoryVectorStore(embeddings)
        vector_tool = create_retriever_tool(
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            name="query_knowledge_base",
            description=(
                "Provides detailed information from documents uploaded by the user. "
                "Always check this tool before answering questions."
            ),
        )
        rag_tools = [vector_tool]

    mcp_tools = await _init_mcp_tools(mcp_config)
    tools = mcp_tools + rag_tools

    if rag_enabled:
        system_prompt = (
            "You are a virtual assistant. Always invoke tool query_knowledge_base before answering questions. "
            "Greet the user and say hello, I am an AI Guard demo agent!!"
        )
    else:
        system_prompt = (
            "You are a virtual assistant. "
            "Greet the user and say hello, I am an AI Guard demo agent!!"
        )

    if provider == "Anthropic":
        if mode == "Proxy" and os.getenv("GUARDRAIL_PROXY_API_KEY"):
            llm = ChatAnthropic(
                model=model,
                base_url="https://proxy.zseclipse.net",
                default_headers={"X-ApiKey": os.getenv("GUARDRAIL_PROXY_API_KEY")},
            )
        else:
            llm = ChatAnthropic(model=model)
    elif provider == "OpenAI":
        if mode == "Proxy" and os.getenv("GUARDRAIL_PROXY_API_KEY"):
            llm = ChatOpenAI(
                model=model,
                base_url="https://proxy.zseclipse.net",
                default_headers={"X-ApiKey": os.getenv("GUARDRAIL_PROXY_API_KEY")},
            )
        else:
            llm = ChatOpenAI(model=model)
    elif provider == "Ollama":
        llm = ChatOllama(model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    checkpointer = InMemorySaver()
    das_active = (
        mode == "DAS"
        and bool(os.getenv("GUARDRAIL_DAS_API_KEY"))
        and bool(os.getenv("GUARDRAIL_DAS_POLICY_ID"))
    )
    if das_active:
        agent = create_agent(
            llm,
            system_prompt=system_prompt,
            tools=tools,
            middleware=[InspectPrompt, InspectResponse],
            checkpointer=checkpointer,
        )
    else:
        agent = create_agent(llm, system_prompt=system_prompt, tools=tools, checkpointer=checkpointer)

    return agent, vector_store


async def chat(
    agent,
    message: str,
    thread_id: str,
    inspect_prompt: bool = True,
    inspect_response: bool = True,
) -> tuple[str, list[dict]]:
    """Send a message to the agent and return (response_text, tool_trace).

    tool_trace is a list of {"name", "input", "output"} dicts, one per tool call.
    """
    inspect_prompt_ctx.set(inspect_prompt)
    inspect_response_ctx.set(inspect_response)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        {"configurable": {"thread_id": thread_id}},
    )

    # Pair each tool_use block with its matching ToolMessage result
    tool_trace: list[dict] = []
    msgs = result["messages"]
    for i, msg in enumerate(msgs):
        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
            continue
        for tc in msg.tool_calls:
            output = None
            for j in range(i + 1, len(msgs)):
                tm = msgs[j]
                if hasattr(tm, "tool_call_id") and tm.tool_call_id == tc["id"]:
                    output = tm.content
                    break
            tool_trace.append({"name": tc["name"], "input": tc["args"], "output": output})

    last = msgs[-1]
    try:
        block = last.content[-1]
        if isinstance(block, dict) and "text" in block:
            return block["text"], tool_trace
    except (IndexError, TypeError):
        pass
    return last.content, tool_trace
