# This is a code sample that shows how AI Guard can be integrated in a LangChain AI Pipeline
# We are inserting AI Guard as part of the chain before the prompt is sent to the LLM and when a response is received
# from the LLM.
##  user prompt --> AI Guard (allow, block, etc)  --> LLM -> AI Guard (allow, block, etc) -->
## Middleware is used to insert AI Guard beforeAgent and after Agent
import asyncio

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import agent_core
from agent_core import build_agent

# --- Session state defaults ---
if "inspect_prompt_enabled" not in st.session_state:
    st.session_state.inspect_prompt_enabled = True

if "inspect_response_enabled" not in st.session_state:
    st.session_state.inspect_response_enabled = True

if "provider" not in st.session_state:
    st.session_state.provider = "Anthropic"

if "model" not in st.session_state:
    st.session_state.model = "claude-haiku-4-5-20251001"

if "mode" not in st.session_state:
    st.session_state.mode = "DAS"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


st.set_page_config(page_title="AI Guard", page_icon="✨")
st.header("AI Guard - Demo agent")
st.divider()

load_dotenv()


@st.cache_resource
def get_agent_anthropic():
    print("Creating agent")
    return build_agent(
        provider=st.session_state.provider,
        model=st.session_state.model,
        mode=st.session_state.mode,
    )


def generate_response(input_text: str):
    answer_box = st.empty()
    try:
        response = asyncio.run(
            agent_core.chat(
                st.session_state.agent,
                input_text,
                thread_id="1",
                inspect_prompt=st.session_state.inspect_prompt_enabled,
                inspect_response=st.session_state.inspect_response_enabled,
            )
        )
    except Exception as e:
        answer_box.markdown(str(e))
        st.session_state["messages"].append({"role": "assistant", "content": str(e)})
        return

    answer_box.markdown(response)
    st.session_state["messages"].append({"role": "assistant", "content": response})


with st.sidebar:
    st.header("AI Guard Inspection")

    newmode = st.radio("Inspection mode:", ["DAS", "Proxy"], key="rdio")

    if newmode != st.session_state.mode:
        st.session_state.mode = newmode
        get_agent_anthropic.clear()
        st.session_state.agent, st.session_state.vectorstore = get_agent_anthropic()

    if st.session_state.mode == "DAS":
        st.session_state.inspect_prompt_enabled = st.toggle(
            "Inspect Prompt (IN)",
            value=st.session_state.inspect_prompt_enabled,
        )
        st.session_state.inspect_response_enabled = st.toggle(
            "Inspect Response (OUT)",
            value=st.session_state.inspect_response_enabled,
        )

    with st.sidebar:
        st.header("Model")
        optionProvider = st.selectbox(
            "Provider",
            ("Anthropic", "Zllama", "Ollama", "OpenAI"),
        )
        if optionProvider == "Anthropic":
            optionModel = st.selectbox(
                "Model",
                ("claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-4-8", "claude-fable-5"),
            )
        if optionProvider == "Ollama":
            optionModel = st.selectbox(
                "Model",
                ("gemma4:e2b"),
            )

        if optionProvider != st.session_state.provider or optionModel != st.session_state.model:
            st.session_state.model = optionModel
            st.session_state.provider = optionProvider
            get_agent_anthropic.clear()
            st.session_state.agent, st.session_state.vectorstore = get_agent_anthropic()

    with st.sidebar:
        st.header("Document Ingestion - RAG Database")
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


# Initialize agent and vector store
st.session_state.agent, st.session_state.vectorstore = get_agent_anthropic()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        generate_response(prompt)
