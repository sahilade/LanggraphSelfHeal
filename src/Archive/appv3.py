import os
import requests
import json
from typing import List, Optional
from pydantic import BaseModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END


class State(BaseModel):
    messages: List[dict] = []


tools = [
    {
        "type": "function",
        "function": {
            "name": "GetAPIEndpoint",
            "description": "Get API Endpoint based on the job type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "string",
                        "description": "Job Type. Can be sap, salesforce or any other."
                    }
                },
                "required": ["x"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_wind",
            "description": "Get wind speed for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
                "additionalProperties": False
            }
        }
    }
]


class InternalLLM:
    def __init__(self, api_url, team_id, project_id, api_key, model_name="gpt-4o-mini", temperature=0.5):
        self.api_url = api_url
        self.team_id = team_id
        self.project_id = project_id
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def invoke(self, messages: List[dict], functions: Optional[List[dict]] = None) -> AIMessage:
        print("\n===============================")
        print("[DEBUG] Invoking Internal LLM with messages:")
        print(json.dumps(messages, indent=2))
        print("===============================")

        payload = {
            "prompt": json.dumps(messages),
            "generation_model": self.model_name,
            "max_tokens": 100,
            "temperature": self.temperature,
            "n": 1,
            "raw_response": True
        }

        if functions:
            payload["tools"] = functions
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {os.getenv('okta_token')}",
            "team_id": self.team_id,
            "project_id": self.project_id,
            "x-pepgenx-apikey": self.api_key,
            "Content-Type": "application/json"
        }

        response = requests.post(
            url=self.api_url,
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        
        choice = response.json()["choices"][0]["message"]
        if choice.get("content") is not None:
            # Normal text response
            return AIMessage(content=choice["content"])
        elif choice.get("tool_calls") is not None:
            # Tool-calling response
            return AIMessage(content="", additional_kwargs={"tool_calls": choice["tool_calls"]})
        else:
            raise Exception(f"Unknown message format from LLM: {choice}")

# -----------------------------
#Load environment variables

from dotenv import load_dotenv

load_dotenv(override=True)

internal_llm = InternalLLM(
    api_url=os.getenv("api_url"),
    team_id=os.getenv("team_id"),
    project_id=os.getenv("project_id"),
    api_key=os.getenv("apikey"),
    model_name="gpt-4o-mini",
    temperature=0.5
)

# -----------------------------
# LangGraph Node - LLM

def llm_node(state: State) -> State:
    messages = state.messages
    response = internal_llm.invoke(messages=messages, functions=tools)
    print("\n[LLM Response] Raw:", response)
    print("[LLM Response] As dict:", response.model_dump())
    return State(messages=messages + [response.model_dump()])

# -----------------------------
# LangGraph Node - User Input

def user_input_node(state: State) -> State:
    print("\n========== Conversation So Far ==========")
    for m in state.messages:
        role = m.get("role")
        if role == "system":
            continue
        elif role == "user":
            print(f">> User: {m.get('content')}")
        elif role == "assistant":
            content = m.get("content")
            if content:
                print(f">> Assistant: {content}")
            elif "tool_calls" in m.get("additional_kwargs", {}):
                for call in m["additional_kwargs"]["tool_calls"]:
                    tool_name = call['function']['name']
                    args = call['function']['arguments']
                    print(f">> Assistant (function call): {tool_name}({args})")
            else:
                print(">> Assistant: [No content or tool_calls]")
    print("=========================================")

    user_text = input("\nYour next message (or type 'exit' to finish): ").strip()
    if user_text.lower() == "exit":
        print("\n[Conversation ended by user]")
        exit(0)

    new_message = {"role": "user", "content": user_text}
    return State(messages=state.messages + [new_message])

# -----------------------------
#  Build LangGraph

graph = StateGraph(State)
graph.add_node("agent", RunnableLambda(llm_node))
graph.add_node("user_input", RunnableLambda(user_input_node))
graph.set_entry_point("agent")
graph.add_edge("agent", "user_input")
graph.add_edge("user_input", "agent")
app = graph.compile()

# -----------------------------
# Execution
initial_state = State(messages=[
    {"role": "system", "content": "You are expert in Indian politics and current affairs."},
    {"role": "user", "content": "How are you doing ?"}

])

app.invoke(initial_state)