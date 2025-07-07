import os
import json
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableLambda
from langchain.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# -------------------------
# Load env variables
load_dotenv()

# -------------------------
# Define the State
class State(BaseModel):
    messages: List[Union[dict, AIMessage, ToolMessage]] = []

# -------------------------
# Mock Tool functions
def get_api_endpoint_func(x: str) -> str:
    """Get API Endpoint based on the job type."""
    return f"https://api.example.com/{x}"

def get_weather_func(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location} is sunny."

def get_wind_func(location: str) -> str:
    """Get wind speed for a location."""
    return f"Wind speed in {location} is 15 km/h."

# -------------------------
# Tools as StructuredTool
tools_list = [
    StructuredTool.from_function(get_api_endpoint_func, name="GetAPIEndpoint"),
    StructuredTool.from_function(get_weather_func, name="get_weather"),
    StructuredTool.from_function(get_wind_func, name="get_wind")
]

tool_node = ToolNode(tools=tools_list)

# -------------------------
# LLM Wrapper
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
            "prompt": messages,
            "generation_model": self.model_name,
            "max_tokens": 500,
            "temperature": self.temperature,
            "n": 1,
            "raw_response": True
        }

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

        choice = response.json()["choices"][0].get("message")
        if choice is None:
            raise Exception("LLM response missing 'message' key")

        # Parse tool calls
        raw_tool_calls = choice.get("tool_calls", [])
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

        content = choice.get("content", "")
        return AIMessage(content=content, tool_calls=tool_calls)

# -------------------------
# Instantiate LLM
internal_llm = InternalLLM(
    api_url=os.getenv("api_url"),
    team_id=os.getenv("team_id"),
    project_id=os.getenv("project_id"),
    api_key=os.getenv("apikey"),
)

# -------------------------
# Agent Node
def agent_node(state: State) -> State:
    response = internal_llm.invoke([m if isinstance(m, dict) else m.dict() for m in state.messages])
    return State(messages=state.messages + [response])

# -------------------------
# Tool Output Node
def tool_output_node(state: State) -> State:
    last_msg = [m for m in state.messages if isinstance(m, AIMessage)][-1]
    tool_messages = []
    for call, result in zip(last_msg.tool_calls, state.tool_results):
        tool_messages.append(
            ToolMessage(
                tool_call_id=call.id,
                content=result
            )
        )
    return State(messages=state.messages + tool_messages)

# -------------------------
# Conditional routing: check if tool_calls exist
def tool_gate(state: State) -> str:
    last_msg = [m for m in state.messages if isinstance(m, AIMessage)][-1]
    return "tools" if last_msg.tool_calls else END

# -------------------------
# LangGraph Flow
graph = StateGraph(State)
graph.add_node("agent", RunnableLambda(agent_node))
graph.add_node("tools", tool_node)
graph.add_node("tool_output", RunnableLambda(tool_output_node))

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tool_gate, {
    "tools": "tools",
    END: END
})
graph.add_edge("tools", "tool_output")
graph.add_edge("tool_output", "agent")

app = graph.compile()

# # -------------------------
# # Run
# initial_state = State(messages=[
#     {"role": "system", "content": "You are an AI assistant."},
#     {"role": "user", "content": "Get me the SAP endpoint."}
# ])

# final_state = app.invoke(initial_state)

# # -------------------------
# # Print final message
# print("\n===== Final Assistant Response =====")
# for msg in final_state.messages:
#     if isinstance(msg, AIMessage):
#         print(f"Assistant: {msg.content}")
