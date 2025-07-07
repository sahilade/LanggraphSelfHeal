import os
import json
import requests
from typing import List, Optional
from dotenv import load_dotenv

from pydantic import BaseModel

# LangChain
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import Tool

# LangGraph
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda
from utils.tools import GetAPIEndpoint, get_weather, get_wind


load_dotenv(override=True)



class State(BaseModel):
    messages: List[BaseMessage] = []
    previous_messages: Optional[List[BaseMessage]] = None

def GetAPIEndpoint(x: str) -> str:
    return f"https://api.example.com/{x}"

def get_weather(location: str) -> str:
    return f"The weather  is sunny 25°C."

def get_wind(location: str) -> str:
    return f"The wind speed in {location} is 10 km/h."

# =====================================================
#  Tool objects
tools = [
    Tool.from_function(
        func=GetAPIEndpoint,
        name="GetAPIEndpoint",
        description="Get API Endpoint based on the job type."
    ),
    Tool.from_function(
        func=get_weather,
        name="get_weather",
        description="Get weather for a location."
    ),
    Tool.from_function(
        func=get_wind,
        name="get_wind",
        description="Get wind speed for a location."
    )
]

tools_schemas = [
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

def serialize_for_endpoint(messages: List[BaseMessage]) -> List[dict]:
    result = []
    print("\n[DEBUG] Serializing messages in Serialize endpoint:")
    print(messages)
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
            role = "tool"
            content = f"[Tool Result: {m.name}] {m.content}"
        else:
            raise ValueError(f"Unsupported message type: {type(m)}")
        result.append({"role": role, "content": content})
    print(result)
    return result

#LLM Object
class InternalLLM:
    def __init__(self, api_url, team_id, project_id, api_key, model_name="gpt-4o-mini", temperature=0.5):
        self.api_url = api_url
        self.team_id = team_id
        self.project_id = project_id
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def invoke(self, messages: List[BaseMessage], functions: Optional[List[dict]] = None) -> AIMessage:
        # Serialize LangChain Message objects to plain dict
        print("*"*50 + "\n[DEBUG] Serializing messages for endpoint:")
        serialized_messages = serialize_for_endpoint(messages)
        
        payload = {
            "prompt": json.dumps(serialized_messages),
            "generation_model": self.model_name,
            "max_tokens": 100,
            "temperature": self.temperature,
            "n": 1,
            "raw_response": True
        }

        if functions:
            # Provide schema to the LLM for function calling
            payload["tools"] = functions
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {os.getenv('okta_token')}",
            "team_id": self.team_id,
            "project_id": self.project_id,
            "x-pepgenx-apikey": self.api_key,
            "Content-Type": "application/json"
        }

        # Make the request

        response = requests.post(self.api_url, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        choice = response.json()["choices"][0]["message"]

        print("\n===============================")
        print("[DEBUG] LLM Response:")
        print( response.json())
        # Build an AIMessage (content and tool_calls both supported)
        
        # return AIMessage(
        #     content=choice.get("content", ""),
        #     additional_kwargs={"tool_calls": choice.get("tool_calls", [])}
        # )

        content = choice.get("content") or ""
        print(f"[DEBUG] Content: {content}")
        tool_calls = choice.get("tool_calls")
        print(f"[DEBUG] Tool calls: {tool_calls}")
        print("AIMSG===============================")
        AIMsg=AIMessage(
            content=content,
            additional_kwargs={"tool_calls": tool_calls} if tool_calls else {}
        )
        print(AIMsg)
        return AIMsg

# =====================================================
# InternalLLM instance
internal_llm = InternalLLM(
    api_url=os.getenv("api_url"),
    team_id=os.getenv("team_id"),
    project_id=os.getenv("project_id"),
    api_key=os.getenv("apikey"),
    model_name="gpt-4o-mini",
    temperature=0.5
)

# =====================================================
# Agent Node
def agent_node(state: State) -> State:
    print("^"*100, "\n[DEBUG] AGENT NODE")
    print(state.messages)
    response = internal_llm.invoke(state.messages, functions=tools_schemas )
    return State(messages=state.messages + [response], previous_messages=state.messages)

# =====================================================
# User Input Node
def user_input_node(state: State) -> State:
    print("\n========== Conversation So Far ==========")
    for m in state.messages:
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, HumanMessage):
            print(f">> User: {m.content}")
        elif isinstance(m, AIMessage):
            print(f">> Assistant: {m}")
            if m.content:
                print(f"**Assistant: {m.content}")
            if "tool_calls" in m.additional_kwargs:
                for call in m.additional_kwargs["tool_calls"]:
                    name = call["function"]["name"]
                    args = call["function"]["arguments"]
                    print(f">> Assistant (function call): {name}({args})")
    print("=========================================")

    user_text = input("\nYour message (or 'exit'): ").strip()
    if user_text.lower() == "exit":
        print("\n[Conversation ended by user]")
        exit(0)

    return State(messages=state.messages + [HumanMessage(content=user_text)])

def tool_output_node(state: State) -> State:
    print("^^"*50, "\n[DEBUG] TOOL OUTPUT NODE")
    return State(messages=state.previous_messages + state.messages)

# =====================================================

tool_node = ToolNode(tools=tools)

def tool_condition(state: State) -> str:
    print("^^"*100, "\n[DEBUG] Checking tool condition")
    last = state.messages[-1]
    print(f"[DEBUG] Last message: {last}")
    if isinstance(last, AIMessage) and last.additional_kwargs.get("tool_calls"):
        print("tools")
        return "tools"
    print("user_input")
    return "user_input"

# =====================================================
# Buiild LangGraph
graph = StateGraph(State)
graph.add_node("agent", RunnableLambda(agent_node))
graph.add_node("tools", tool_node)
graph.add_node("user_input", RunnableLambda(user_input_node))
graph.add_node("tool_output", RunnableLambda(tool_output_node))

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    tool_condition,
    {
        "tools": "tools",
        "user_input": "user_input"
    }
)
graph.add_edge("tools", "tool_output")
graph.add_edge("tool_output", "agent")

graph.add_edge("user_input", "agent")

app = graph.compile()

# =====================================================

initial_state = State(messages=[
    SystemMessage(content="You are expert in Indian politics and current affairs."),
    HumanMessage(content="How are you doing?")
])

app.invoke(initial_state)
