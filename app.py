# This is a code sample that shows how AI Guard can be integrated in a LangChain AI Pipeline
# We are inserting AI Guard as part of the chain before the prompt is sent to the LLM and when a response is received
# from the LLM.
##  user prompt --> AI Guard (allow, block, etc)  --> LLM -> AI Guard (allow, block, etc) -->
## Middleware is used to insert AI Guard beforeAgent and after Agent
import asyncio
import json
import os
import uuid

import streamlit as st
from dotenv import find_dotenv, load_dotenv, set_key
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import agent_core
from agent_core import DEFAULT_MCP_CONFIG, build_agent

# --- Session state defaults ---
if "inspect_prompt_enabled" not in st.session_state:
    st.session_state.inspect_prompt_enabled = True

if "inspect_response_enabled" not in st.session_state:
    st.session_state.inspect_response_enabled = True

# Provider availability:
#  - Anthropic / OpenAI require their API keys to be configured.
#  - Ollama requires a reachable local Ollama server.
# Probe Ollama once per session and reuse the result for provider selection
# and the RAG (document ingestion) UI.
_anthropic_enabled = bool(os.getenv("ANTHROPIC_API_KEY"))
_openai_enabled = bool(os.getenv("OPENAI_API_KEY"))

if "ollama_available" not in st.session_state:
    st.session_state.ollama_available = agent_core.ollama_available()
_ollama_enabled = st.session_state.ollama_available

# Cache the list of installed Ollama models for the Model selector.
if "ollama_models" not in st.session_state:
    st.session_state.ollama_models = (
        agent_core.list_ollama_models() if _ollama_enabled else []
    )

# Cache the list of available Anthropic models for the Model selector.
if "anthropic_models" not in st.session_state:
    st.session_state.anthropic_models = (
        agent_core.list_anthropic_models() if _anthropic_enabled else []
    )

# Cache the list of available OpenAI models for the Model selector.
if "openai_models" not in st.session_state:
    st.session_state.openai_models = (
        agent_core.list_openai_models() if _openai_enabled else []
    )

if "provider" not in st.session_state:
    if _anthropic_enabled:
        st.session_state.provider = "Anthropic"
    elif _openai_enabled:
        st.session_state.provider = "OpenAI"
    else:
        st.session_state.provider = "Ollama"

if "model" not in st.session_state:
    if _anthropic_enabled:
        st.session_state.model = (
            st.session_state.anthropic_models[0]
            if st.session_state.anthropic_models
            else "claude-haiku-4-5-20251001"
        )
    elif _openai_enabled:
        st.session_state.model = (
            st.session_state.openai_models[0]
            if st.session_state.openai_models
            else "gpt-4o"
        )
    else:
        st.session_state.model = (
            st.session_state.ollama_models[0]
            if st.session_state.ollama_models
            else "gemma4:e2b"
        )

if "mode" not in st.session_state:
    _available = agent_core.available_modes()
    st.session_state.mode = _available[0] if _available else "off"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "mcp_tools_config" not in st.session_state:
    st.session_state.mcp_tools_config = dict(DEFAULT_MCP_CONFIG)

if "agent" not in st.session_state:
    st.session_state.agent = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# One persistent event loop per session. asyncio.run() closes the loop after each
# call, which invalidates the httpx.AsyncClient held inside ChatAnthropic and the
# MCP client connections. Reusing the same loop keeps those resources alive across
# multiple chat turns.
if "event_loop" not in st.session_state:
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    st.session_state.event_loop = _loop


def run_async(coro):
    return st.session_state.event_loop.run_until_complete(coro)


st.set_page_config(page_title="AI Guard", page_icon="✨")
st.header("AI Guard - Demo agent")
##st.divider()

load_dotenv()


def reset_agent():
    """Recreate the agent for the current provider/model/mode and start a fresh conversation."""
    st.session_state.agent, st.session_state.vectorstore = run_async(
        build_agent(
            provider=st.session_state.provider,
            model=st.session_state.model,
            mode=st.session_state.mode,
            mcp_config=st.session_state.mcp_tools_config,
        )
    )
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())


def _render_tool_calls(tool_calls: list[dict]):
    for tc in tool_calls:
        st.markdown(f"**`{tc['name']}`**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Input")
            st.json(tc["input"])
        with col2:
            st.caption("Output")
            output = tc["output"]
            if output is None:
                st.caption("—")
            elif isinstance(output, str):
                st.text(output[:2000])
            else:
                st.json(output)
        st.divider()


def generate_response(input_text: str):
    answer_box = st.empty()
    try:
        response, tool_trace = run_async(
            agent_core.chat(
                st.session_state.agent,
                input_text,
                thread_id=st.session_state.thread_id,
                inspect_prompt=st.session_state.inspect_prompt_enabled,
                inspect_response=st.session_state.inspect_response_enabled,
            )
        )
    except Exception as e:
        answer_box.markdown(str(e))
        st.session_state["messages"].append({"role": "assistant", "content": str(e), "tool_calls": []})
        return

    answer_box.markdown(response)
    if tool_trace:
        with st.expander(f"🔧 Tool calls ({len(tool_trace)})"):
            _render_tool_calls(tool_trace)
    st.session_state["messages"].append({"role": "assistant", "content": response, "tool_calls": tool_trace})


with st.sidebar:


    modes = agent_core.available_modes()

    if not modes:
        st.caption("No inspection modes available — configure GUARDRAIL keys below.")
        if st.session_state.mode != "off":
            st.session_state.mode = "off"
            reset_agent()
    else:
        # If the current mode was removed (e.g. key deleted), fall back gracefully
        if st.session_state.mode not in modes:
            st.session_state.mode = modes[0]
            reset_agent()

        newmode = st.radio(
            "AI Guard Inspection mode:", modes,
            index=modes.index(st.session_state.mode),
        )
        if newmode != st.session_state.mode:
            st.session_state.mode = newmode
            reset_agent()

        if st.session_state.mode == "DAS":
            st.session_state.inspect_prompt_enabled = st.toggle(
                "Inspect Prompt (IN)",
                value=st.session_state.inspect_prompt_enabled,
            )
            st.session_state.inspect_response_enabled = st.toggle(
                "Inspect Response (OUT)",
                value=st.session_state.inspect_response_enabled,
            )

    with st.expander("AI Guard Configuration"):


        guardrail_vars = {k: v for k, v in os.environ.items() if k.startswith("GUARDRAIL_")}

        if not guardrail_vars:
            st.caption("No GUARDRAIL_* variables found in environment.")
        else:
            with st.form("guardrail_config"):
                new_values = {}
                for var_name in sorted(guardrail_vars):
                    label = var_name.removeprefix("GUARDRAIL_").replace("_", " ").title()
                    is_secret = any(var_name.endswith(s) for s in ("_KEY", "_TOKEN", "_SECRET"))
                    new_values[var_name] = st.text_input(
                        label,
                        value=guardrail_vars[var_name],
                        type="password" if is_secret else "default",
                    )

                if st.form_submit_button("💾 Save & Apply"):
                    dotenv_path = find_dotenv(usecwd=True)
                    changed = any(
                        new_val != os.environ.get(var_name, "")
                        for var_name, new_val in new_values.items()
                    )
                    if changed:
                        for var_name, new_val in new_values.items():
                            os.environ[var_name] = new_val
                            if dotenv_path:
                                set_key(dotenv_path, var_name, new_val)
                        agent_core.reconfigure_guard()
                        if dotenv_path:
                            st.success("Saved to .env and applied.")
                        else:
                            st.warning("Applied for this session — no .env file found, changes won't survive a restart.")
                    else:
                        st.info("No changes detected.")

    with st.expander("Model"):

        # Only offer Anthropic / OpenAI when their API keys are configured,
        # and Ollama only when a local Ollama server is reachable.
        provider_options = []
        if os.getenv("ANTHROPIC_API_KEY"):
            provider_options.append("Anthropic")
        if os.getenv("OPENAI_API_KEY"):
            provider_options.append("OpenAI")
        if st.session_state.ollama_available:
            provider_options.append("Ollama")

        optionProvider = st.selectbox(
            "Provider",
            provider_options,
        )
        if optionProvider == "Anthropic":
            optionModel = st.selectbox(
                "Model",
                st.session_state.anthropic_models,
            )
        elif optionProvider == "OpenAI":
            optionModel = st.selectbox(
                "Model",
                st.session_state.openai_models,
            )
        elif optionProvider == "Ollama":
            if st.session_state.ollama_models:
                optionModel = st.selectbox(
                    "Model",
                    st.session_state.ollama_models,
                )
            else:
                st.warning("No Ollama models installed. Pull one with `ollama pull <model>`.")
                optionModel = st.session_state.model
        else:
            st.warning(f"Provider '{optionProvider}' is not supported yet.")
            optionModel = st.session_state.model

        if optionProvider != st.session_state.provider or optionModel != st.session_state.model:
            st.session_state.provider = optionProvider
            st.session_state.model = optionModel
            reset_agent()

    with st.expander("MCP Tools"):


        if not st.session_state.mcp_tools_config:
            st.caption("No MCP tools configured.")

        for tool_name, tool_settings in list(st.session_state.mcp_tools_config.items()):
            with st.expander(f"🔧 {tool_name}"):
                st.text_input("URL", value=tool_settings["url"],
                              key=f"disp_url_{tool_name}", disabled=True)
                headers = tool_settings.get("headers") or {}
                st.text_area("Headers (JSON)", height=80,
                             value=json.dumps(headers, indent=2) if headers else "",
                             key=f"disp_hdr_{tool_name}", disabled=True,
                             help="HTTP headers sent with every request to this tool")
                if st.button("Remove", key=f"remove_mcp_{tool_name}", type="secondary"):
                    del st.session_state.mcp_tools_config[tool_name]
                    reset_agent()
                    st.rerun()

        # Add new tool form
        with st.form("add_mcp_tool", clear_on_submit=True):
            st.markdown("**Add Tool**")
            new_name = st.text_input("Name", placeholder="e.g. weather")
            new_url = st.text_input("URL", placeholder="https://...")
            new_headers_raw = st.text_area(
                "Headers (JSON, optional)", height=80,
                placeholder='{"Authorization": "Bearer <token>"}',
                help="Optional HTTP headers to include with every request to this tool",
            )
            if st.form_submit_button("Add"):
                if new_name and new_url:
                    headers = {}
                    if new_headers_raw.strip():
                        try:
                            headers = json.loads(new_headers_raw)
                            if not isinstance(headers, dict):
                                st.error("Headers must be a JSON object, e.g. {\"key\": \"value\"}")
                                st.stop()
                        except json.JSONDecodeError:
                            st.error("Invalid JSON — headers must be a JSON object")
                            st.stop()
                    st.session_state.mcp_tools_config[new_name] = {
                        "url": new_url,
                        "headers": headers,
                    }
                    reset_agent()
                    st.rerun()

    # RAG document ingestion needs Ollama (nomic-embed-text) for embeddings.
    # Only show this section when Ollama is reachable.
    if st.session_state.ollama_available:
        with st.expander("Document Ingestion - Local RAG DB"):

            uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])
            if uploaded_file is not None:
                if st.button("Process & Build Knowledge Base", type="primary"):
                    with st.spinner("Parsing file and building vector index..."):
                        try:
                            text_content = uploaded_file.read().decode("utf-8")
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                            chunks = text_splitter.split_text(text_content)
                            documents = [
                                Document(page_content=chunk, metadata={"source": uploaded_file.name})
                                for chunk in chunks
                            ]
                            if documents:
                                st.session_state.vectorstore.add_documents(documents)
                                print("Successfully indexed the document")
                        except Exception as e:
                            st.error(f"Processing error {e}")


# Initialize agent on first load
if st.session_state.agent is None:
    st.session_state.agent, st.session_state.vectorstore = run_async(
        build_agent(
            provider=st.session_state.provider,
            model=st.session_state.model,
            mode=st.session_state.mode,
            mcp_config=st.session_state.mcp_tools_config,
        )
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tool_calls"):
            with st.expander(f"🔧 Tool calls ({len(message['tool_calls'])})"):
                _render_tool_calls(message["tool_calls"])

# React to user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        generate_response(prompt)
