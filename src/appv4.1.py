import os
import requests
import json
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ToolCall,
)
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

# -----------------------------
# Environment
load_dotenv(override=True)

# -----------------------------
# State
class State(BaseModel):
    messages: List = []

# -----------------------------
# Tools
tools_schemas = [
    {
        "type": "function",
        "function": {
            "name": "GetAPIEndpoint",
            "description": "Get API Endpoint based on the job type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "string", "description": "Job Type. Can be sap, salesforce or any other."}
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

# -----------------------------
# Helper: serialize history for your endpoint
def serialize_for_endpoint(messages: List) -> List[dict]:
    """
    CHANGE: Added ToolMessage support
    This ensures ToolMessage is serialized as an assistant response.
    """
    result = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "user"
            content = m.content
        elif isinstance(m, AIMessage):
            role = "assistant"
            content = m.content
        elif isinstance(m, SystemMessage):
            role = "system"
            content = m.content
        elif isinstance(m, ToolMessage):
            # Tool results - treat as assistant output
            role = "assistant"
            content = m.content
        else:
            raise ValueError(f"Unsupported message type: {type(m)}")
        result.append({"role": role, "content": content})
    return result

# -----------------------------
# Internal LLM
class InternalLLM:
    def __init__(self, api_url, team_id, project_id, api_key, model_name="gpt-4o-mini", temperature=0.5):
        self.api_url = api_url
        self.team_id = team_id
        self.project_id = project_id
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def invoke(self, messages: List, functions: Optional[List[dict]] = None) -> AIMessage:
        """
        CHANGE: serialize_for_endpoint ensures conversation history is in endpoint-accepted format
        """
        serialized_messages = serialize_for_endpoint(messages)

        payload = {
            "prompt": json.dumps(serialized_messages),
            "generation_model": self.model_name,
            "max_tokens": 300,
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
        print("\n===============================")
        print("[DEBUG] LLM Response:")
        print(json.dumps(choice, indent=2))
        print("===============================")

        # -----------------------------
        # CHANGE: Parse tool_calls from raw JSON
        raw_tool_calls = choice.get("tool_calls")
        tool_calls = []
        if raw_tool_calls:
            tool_calls = []
            for raw in raw_tool_calls:
                arguments = json.loads(raw["function"]["arguments"])
                tool_calls.append(
                    ToolCall(
                        id=raw["id"],
                        name=raw["function"]["name"],
                        arguments=arguments
                    )
                )

        content = choice.get("content") or ""

        # -----------------------------
        # CHANGE: Return AIMessage with tool_calls
        return AIMessage(
            content=content,
            tool_calls=tool_calls
        )

# -----------------------------
# Initialize LLM
internal_llm = InternalLLM(
    api_url=os.getenv("api_url"),
    team_id=os.getenv("team_id"),
    project_id=os.getenv("project_id"),
    api_key=os.getenv("apikey"),
    model_name="gpt-4o-mini",
    temperature=0.5
)

# -----------------------------
# Graph Node - Agent
def agent_node(state: State) -> State:
    """
    CHANGE: agent_node now always returns AIMessage with tool_calls (if any)
    enabling automatic routing to ToolNode.
    """
    response = internal_llm.invoke(state.messages, functions=tools_schemas)
    print("\n[LLM Response] As object:", response)
    return State(messages=state.messages + [response])

# -----------------------------
# Graph Node - User Input
def user_input_node(state: State) -> State:
    print("\n========== Conversation So Far ==========")
    for m in state.messages:
        if isinstance(m, SystemMessage):
            continue
        elif isinstance(m, HumanMessage):
            print(f">> User: {m.content}")
        elif isinstance(m, AIMessage):
            if m.content:
                print(f">> Assistant: {m.content}")
            elif m.tool_calls:
                for call in m.tool_calls:
                    print(f">> Assistant (function call): {call.name}({call.arguments})")
        elif isinstance(m, ToolMessage):
            print(f">> Tool Result: {m.content}")
    print("=========================================")

    user_text = input("\nYour next message (or type 'exit' to finish): ").strip()
    if user_text.lower() == "exit":
        print("\n[Conversation ended by user]")
        exit(0)

    new_message = HumanMessage(content=user_text)
    return State(messages=state.messages + [new_message])

# -----------------------------
# Graph Node - Tools
from langgraph.prebuilt import ToolNode

# Define Python-side functions for the tools
def get_api_endpoint_func(x: str) -> str:
    """Get API Endpoint based on the job type."""
    return f"https://api.example.com/{x}"

def get_weather_func(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location} is sunny."

def get_wind_func(location: str) -> str:
    """Get wind speed for a location."""
    return f"Wind speed in {location} is 15 km/h."

# -----------------------------
from langchain.tools import StructuredTool

tool_map = [
    StructuredTool.from_function(get_api_endpoint_func, name="GetAPIEndpoint"),
    StructuredTool.from_function(get_weather_func, name="get_weather"),
    StructuredTool.from_function(get_wind_func, name="get_wind"),
]

tool_node = ToolNode(
    tools=tool_map
)

# -----------------------------
# Build LangGraph
graph = StateGraph(State)
graph.add_node("agent", RunnableLambda(agent_node))
graph.add_node("user_input", RunnableLambda(user_input_node))
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

# -----------------------------
# CHANGE: Conditional routing based on presence of tool_calls
graph.add_conditional_edges(
    "agent",
    lambda state: "tools" if state.messages[-1].tool_calls else "user_input"
)

graph.add_edge("tools", "agent")

app = graph.compile()

# -----------------------------
# Execution
initial_state = State(messages=[
    SystemMessage(content="You are expert in Indian politics and current affairs."),
    HumanMessage(content="How are you doing ?")
])

app.invoke(initial_state)
