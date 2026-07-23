# This is a code sample that shows how AI Guard can be integrated in a LangChain AI Pipeline
# We are inserting AI Guard as part of the chain before the prompt is sent to the LLM and when a response is received
# from the LLM.
##  user prompt --> AI Guard (allow, block, etc)  --> LLM -> AI Guard (allow, block, etc) -->
##

import threading, time
import queue
import logging
from aiguard_utils import AIGuardClient

from langchain_anthropic import ChatAnthropic
from streamlit.runtime.scriptrunner import add_script_run_ctx
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState,before_agent, after_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain.messages import AIMessage,ToolMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.callbacks import CallbackManagerForToolRun
from typing import Any, Dict, Optional, Tuple
import os
import streamlit as st
import langchain_anthropic


from dotenv import load_dotenv

from datasets import load_dataset



st.set_page_config(page_title="AI Guard - Test assistant", page_icon="✨")

load_dotenv()

class QueueToolLogger(BaseCallbackHandler):
    def __init__(self, q: "queue.Queue[str]"):
        self.q = q

    def on_text(self, text: str, **kwargs):
        self.q.put(text)

    def on_tool_start(self, serialized, input_str=None, **kwargs):
        name = (serialized or {}).get("name", "tool")
        self.q.put(f"\n---\nStarting {name}...\n")

    def on_tool_end(self, output, **kwargs):
        if isinstance(output, dict) and "results" in output:
            total = len(output["results"])
            failed = sum(1 for r in output["results"] if r.get("result") == "failed")
            passed = total - failed
            self.q.put(
                f"\nFinished tool. total={total}, passed={passed}, failed={failed}\n"
            )
        else:
            self.q.put(f"\nFinished tool. output={output}\n")

    def on_tool_error(self, error, **kwargs):
        self.q.put(f"\nTool failed: {error}\n")



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
            "messages": [("Traffic was blocked by Zscaler AI Guard")],
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


##@tool
##def AdjustSensitivityLevel(string level)
##    """Adjust the detection sensitiivty level based on the level passed, level can either be high, medium or low"""




@tool
def testPromptInjectionPrompts(_: str = "", run_manager: Optional[CallbackManagerForToolRun] = None) -> dict:
##def testPromptInjectionPrompts(run_manager: Optional[CallbackManagerForToolRun] = None) -> dict:
    """Runs prompt injection tests against Zscaler AI Guard and outputs a list of prompts that failed identified by failed:
       Returns a dictonary containing the results of the test
    """

    def log(s: str):
        if run_manager:
            run_manager.on_text(s + "\n",verbose=True)
        else:
            print("run manager is not configured")
            print(s)

    log("Started testing prompt injection against AI Guard")
    i = 0
    results = {"results": []}

    print("Started testing prompt injection against AI Guard")
    ds = load_dataset("qualifire/prompt-injections-benchmark", token="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")
    for d in ds['test']:
        ##only send prompts labelled as 'jailbreak'
        if d['label'] == 'jailbreak':
            try:
                i = i + 1
                ##if i>10:
                ##    return results
                array = guard.enforce(direction="IN", content=d['text'])
                ##result += "The following test failed:"
                ##result+= d['text']
                ##result+="\r\n"
                results['results'].append({'test': d['text'],'result': 'failed'})
                log(f"[{i}] failed: prompt was allowed by guard")
                ##st.session_state.messages.append({"role":"assistant","content":"TEST FAILED|"+d['text']})

            except ValueError as e:
                results['results'].append({'test': d['text'], 'result': 'passed'})
                if i % 10 == 0:
                    passed = sum(1 for r in results["results"] if r["result"] == "passed")
                    failed = len(results["results"]) - passed
                    log(f"Progress: tested={len(results['results'])}, passed={passed}, failed={failed}")
                ##result+="The following test succeeded:"
                ##result+= d['text']
                ##result += "\r\n"
                ##st.session_state.messages.append({"role": "assistant", "content": "TEST SUCCEEDED|" + d['text']})
                continue

    ##print(results)
    passed = sum(1 for r in results["results"] if r["result"] == "passed")
    failed = len(results["results"]) - passed
    log(f"Completed prompt injection tests. tested={len(results['results'])}, passed={passed}, failed={failed}")
    return results

@tool()
def inspect(promptOrResponse):
    """
    Inspect tool: inspects every prompt and response
    parameter: promptOrResponse either the prompt or response

    :return:
    """
    print("inspecting:" + promptOrResponse)
    return

@tool()
def testToxicity() -> str:
    """Runs toxicity tests against Zscaler AI Guard
       Returns a string containing the results of the test
    """
    result = ""
    i=0

    ds = load_dataset("allenai/real-toxicity-prompts", token="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")
    for d in ds['train']:
        ##only send prompts labelled as 'jailbreak'
        ##only test with high toxicity records
        if d['prompt']['toxicity']!=None and float(d['prompt']['toxicity'])>0.8:

            print("toxicity:")
            print(d['prompt']['toxicity'])
            print(d['prompt']['text'])
            try:

                ##print(d['prompt'])
                array = guard.enforce(direction="IN", content=d['prompt']['text'])
                result+="The following test failed: "
                result+=(d['prompt']['text'])
                result+="\r\n"
                ##test the first 50 entries
                if i>50:
                    break
            except ValueError as e:

                ##result+="The following test succeeded: "
                ##result+=(d['prompt']['text'])
                ##result+="\r\n"

                continue

            i = i + 1
    return result





@st.cache_resource
def get_agent():
    print("Creating agent")
    ##model=gpt-4.1
    ##systemprompt="You are a virtual assistant. You have access to a tool named inspect that must be called on every user message and on every assistant message you produce."
    ##ChatAnthropic(base_url="https://proxy.zseclipse.com",default_headers={"X-ApiKey": ""})

    return create_agent(model="anthropic:claude-sonnet-4-5-20250929",tools=[inspect,testPromptInjectionPrompts,testToxicity], middleware=[InspectPrompt, InspectResponse],checkpointer=InMemorySaver())

agent=get_agent()



def generate_response(input_text: str):
    tool_box = st.empty()
    answer_box = st.empty()

    log_q: queue.Queue[str] = queue.Queue()
    cb = QueueToolLogger(log_q)

    result_holder = {"result": None, "error": None}

    def run_agent():
        try:
            result_holder["result"] = agent.invoke(
                {"messages": [{"role": "user", "content": input_text}]},
                {"configurable": {"thread_id": "1"}, "callbacks": [cb]},
            )
        except Exception as e:
            result_holder["error"] = e

    t = threading.Thread(target=run_agent, daemon=True)
    add_script_run_ctx(t)  # attach Streamlit session context to the worker thread
    t.start()

    tool_log = ""
    while t.is_alive() or not log_q.empty():
        while not log_q.empty():
            tool_log += log_q.get_nowait()
        if tool_log:
            tool_box.markdown(f"### Tool output (live)\n```text\n{tool_log}\n```")
        time.sleep(0.05)

    if result_holder["error"]:
        raise result_holder["error"]

    result = result_holder["result"]
    last_message = result["messages"][-1]
    answer_box.markdown(last_message.content)

    # store into history exactly once
    st.session_state["messages"].append({"role": "assistant", "content": last_message.content})



# --- Inspection toggles (UI) ---
if "inspect_prompt_enabled" not in st.session_state:
    st.session_state.inspect_prompt_enabled = True

if "inspect_response_enabled" not in st.session_state:
    st.session_state.inspect_response_enabled = True

with st.sidebar:
    st.header("AI Guard Inspection")
    st.session_state.inspect_prompt_enabled = st.toggle(
        "Inspect Prompt (IN)",
        value=st.session_state.inspect_prompt_enabled,
    )
    st.session_state.inspect_response_enabled = st.toggle(
        "Inspect Response (OUT)",
        value=st.session_state.inspect_response_enabled,
    )




# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

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



