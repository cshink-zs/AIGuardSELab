import asyncio
import contextvars
import os
from typing import Any

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
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

load_dotenv()

# Per-request inspection flags — set by chat() before invoking the agent
inspect_prompt_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar("inspect_prompt", default=True)
inspect_response_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar("inspect_response", default=True)

guard = AIGuardClient(
    bearer_token=os.getenv("GUARDRAIL_API_KEY"),
    policy_id=os.getenv("GUARDRAIL_POLICY_ID"),
)


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
        return {"messages": [AIMessage(str(e))], "jump_to": "end"}
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
        last.content = str(e)
    return None


async def _init_mcp_tools():
    client = MultiServerMCPClient(
        {"dlptest": {"transport": "http", "url": "https://mcp.dlptest.com/api/mcp/"}}
    )
    return await client.get_tools()


def build_agent(provider: str, model: str, mode: str):
    """Create an agent and vector store for the given provider/model/mode."""
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

    mcp_tools = asyncio.run(_init_mcp_tools())
    tools = mcp_tools + [vector_tool]

    system_prompt = (
        "You are a virtual assistant. Always invoke query_knowledge_base before answering questions. "
        "Greet the user and say hello, I am ZAgent!!"
    )

    if provider == "Anthropic":
        if mode == "Proxy":
            llm = ChatAnthropic(
                model=model,
                base_url="https://proxy.zseclipse.net",
                default_headers={"X-ApiKey": os.getenv("GUARDRAIL_PROXY_API_KEY")},
            )
        else:
            llm = ChatAnthropic(model=model)
    elif provider == "Ollama":
        llm = ChatOllama(model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    checkpointer = InMemorySaver()
    if mode == "Proxy":
        agent = create_agent(llm, system_prompt=system_prompt, tools=tools, checkpointer=checkpointer)
    else:
        agent = create_agent(
            llm,
            system_prompt=system_prompt,
            tools=tools,
            middleware=[InspectPrompt, InspectResponse],
            checkpointer=checkpointer,
        )

    return agent, vector_store


async def chat(
    agent,
    message: str,
    thread_id: str,
    inspect_prompt: bool = True,
    inspect_response: bool = True,
) -> str:
    """Send a message to the agent and return the response text."""
    inspect_prompt_ctx.set(inspect_prompt)
    inspect_response_ctx.set(inspect_response)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        {"configurable": {"thread_id": thread_id}},
    )

    last = result["messages"][-1]
    try:
        block = last.content[-1]
        if isinstance(block, dict) and "text" in block:
            return block["text"]
    except (IndexError, TypeError):
        pass
    return last.content
