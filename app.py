# This is a code sample that shows how AI Guard can be integrated in a LangChain AI Pipeline
# We are inserting AI Guard as part of the chain before the prompt is sent to the LLM and when a response is received
# from the LLM.
##  user prompt --> AI Guard (allow, block, etc)  --> LLM -> AI Guard (allow, block, etc) -->
## Middleware is used to insert AI Guard beforeAgent and after Agent

import threading, time

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from aiguard_utils import AIGuardClient

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState,before_agent, after_agent
from langchain.messages import AIMessage,ToolMessage
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from typing import Any, Dict, Optional, Tuple
import os
import streamlit as st
from langchain_ollama import ChatOllama

##RAG##
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

##END RAG##

##import langchain_anthropic



from dotenv import load_dotenv

from datasets import load_dataset



st.set_page_config(page_title="AI Guard - Test assistant", page_icon="✨")

load_dotenv()


api_key = os.getenv("GUARDRAIL_API_KEY")
##global variables initialized
guard = AIGuardClient(
    bearer_token=api_key,
    policy_id="1190",
)

@before_agent(can_jump_to=["end"])
def InspectPrompt(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:

    # Toggle: skip prompt inspection
    if not st.session_state.get("inspect_prompt_enabled", True):
        return None


    returnValue = None


    try:
        if not state["messages"]:
            return None

        first_message = state["messages"][-1]
        if first_message.type != 'human':
            return None




        print("Prompt inspected: ")
        print(first_message.content)
        guard.enforce(direction="IN", content=first_message.content)


    except ValueError as e:
       returnValue = {
            "messages": [AIMessage("Traffic was blocked by Zscaler AI Guard")],
            "jump_to": "end"
        }
       return returnValue


    return returnValue

@after_agent(can_jump_to=["end"])
def InspectResponse(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Model-based guardrail: Use an LLM to evaluate response safety."""
    # Toggle: skip response inspection
    if not st.session_state.get("inspect_response_enabled", True):
        return None

    # Get the final AI response
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]
    if last_message.type != 'ai':
        return None

    ##skip the inspection of the response as the AI Message was generated during the inspection
    if last_message.content.startswith("Traffic was blocked by Zscaler AI Guard"):
        return None

    try:
        print("Response inspected: ")
        print(last_message.content)
        guard.enforce(direction="OUT", content=last_message.content)

    except ValueError as e:
        last_message.content = "Response was blocked by Zscaler AI Guard"

    return None




@st.cache_resource
def get_agent():
    print("Creating agent")
    ##model=gpt-4.1
    ##systemprompt="You are a virtual assistant. You have access to a tool named inspect that must be called on every user message and on every assistant message you produce."
    ##ChatAnthropic(base_url="https://proxy.zseclipse.com",default_headers={"X-ApiKey": ""})

    # 1. Initialize your local Ollama model
    llmolama = ChatOllama(model="gemma4:e2b")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store=InMemoryVectorStore(embeddings)
    vector_tool = create_retriever_tool(
        retriever=vector_store.as_retriever(search_kwargs={"k":2}),
        name="query_knowledge_base",
        description="Provides detailed information from documents uploaded by the user. "
                    "Always check this tool before answering questions about specific reports or data."
    )
    return create_agent(llmolama,tools=[vector_tool], middleware=[InspectPrompt, InspectResponse],checkpointer=InMemorySaver()),vector_store

agent,vector_store=get_agent()




def generate_response(input_text: str):
        answer_box = st.empty()

        result = agent.invoke(
            {"messages": [{"role": "user", "content": input_text}]},
            {"configurable": {"thread_id": "1"}},
        )

        last_message = result["messages"][-1]
        answer_box.markdown(last_message.content)

        # Store into history exactly once
        st.session_state["messages"].append({"role": "assistant", "content": last_message.content})



# --- Inspection toggles (UI) ---
if "inspect_prompt_enabled" not in st.session_state:
    st.session_state.inspect_prompt_enabled = True

if "inspect_response_enabled" not in st.session_state:
    st.session_state.inspect_response_enabled = True

with (st.sidebar):
    st.header("AI Guard Inspection")
    st.session_state.inspect_prompt_enabled = st.toggle(
        "Inspect Prompt (IN)",
        value=st.session_state.inspect_prompt_enabled,
    )
    st.session_state.inspect_response_enabled = st.toggle(
        "Inspect Response (OUT)",
        value=st.session_state.inspect_response_enabled,
    )

    with st.sidebar:
        st.header("Document Ingestion - RAG Database")
        uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])
        if uploaded_file is not None:
            if st.button("Process & Build Knowledge Base", type="primary"):
                with st.spinner("Parsing file and building vector vector index..."):
                    try:
                        text_content = uploaded_file.read().decode("utf-8")
                        # --- Chunk and Tokenize Text ---
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        chunks = text_splitter.split_text(text_content)
                        # Convert chunks into LangChain Document objects
                        documents = [
                            Document(page_content=chunk, metadata={"source": uploaded_file.name})
                            for chunk in chunks
                        ]
                        if documents:
                            vector_store.add_documents(documents)
                            print("Successfully indexed the document")

                        # 3. Query the Vector Store (to test it works)
                       ## query = st.text_input("Ask a question about your uploaded documents:")
                       ## if query is not None:
                       ##     print("searching for "+query)
                            # Perform similarity search against the session-stored vector database
                       ##     results = vector_store.similarity_search("KevinRoberts@hotmail.com", k=1)

                       ##     st.write("### Most Similar Chunks Found:")
                       ##     for i, doc in enumerate(results):
                       ##         st.info(
                       ##             f"**Chunk {i + 1}** (Source: {doc.metadata.get('source')}):\n\n{doc.page_content}")

                        ##vector_store
                    except Exception as e:
                        st.error(f"Processing error {e}")



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
   ## st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
      st.markdown(prompt)
      ##generate_response(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        generate_response(prompt)
        ##st.markdown(st.session_state.messages[-1]["content"])



