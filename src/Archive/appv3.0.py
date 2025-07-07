import os
import requests
import json
from typing import List, Optional, TypedDict
from pydantic import BaseModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END


# -----------------------------
# ✅ Define your State for LangGraph
# -----------------------------
class State(BaseModel):
    messages: List[dict] = []


# -----------------------------
# ✅ Define your tools
# (These are passed as OpenAI-style JSON definitions)
# -----------------------------
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


# -----------------------------
# ✅ Your InternalLLM calling your custom internal backend
# -----------------------------
class InternalLLM:
    def __init__(self, api_url, team_id, project_id, api_key, model_name="gpt-4o-mini", temperature=0.5):
        self.api_url = api_url
        self.team_id = team_id
        self.project_id = project_id
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def invoke(self, messages: List[dict], functions: Optional[List[dict]] = None) -> AIMessage:
        print("***********************************************************************")
        print("Invoking Internal LLM with messages:", messages, type(messages))
        print("JSON Dumps with messages:", json.dumps(messages, indent=2))
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
        
        choice  = response.json()["choices"][0]["message"]
        if choice.get("content") is not None:
            # Normal text response
            return AIMessage(content=choice["content"])

        elif choice.get("tool_calls") is not None:
            # Tool-calling response
            return AIMessage(content="", additional_kwargs={"tool_calls": choice["tool_calls"]})

        else:
            raise Exception(f"Unknown message format from LLM: {choice}")

# -----------------------------
# ✅ Load your .env or hardcode if you want
# -----------------------------
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
# ✅ LangGraph node that calls your backend
# -----------------------------
def llm_node(state: State) -> State:
    messages = state.messages
    response = internal_llm.invoke(messages=messages, functions=tools)
    print("response from internal_llm:", response)
    print("model_dump from internal_llm:", response.model_dump())
    return State(messages=messages + [response.model_dump()])




# -----------------------------
# ✅ Build LangGraph
# -----------------------------
graph = StateGraph(State)
graph.add_node("agent", RunnableLambda(llm_node))
graph.add_node("node2", RunnableLambda(node2))
graph.set_entry_point("agent")
graph.add_edge("agent", "node2")
graph.add_edge("node2", END)
app = graph.compile()


# -----------------------------
# ✅ Run it
# -----------------------------
initial_state = State(messages=[
    {"role": "system", "content": "You are expert in Indian politics and current affairs."},
    {"role": "user", "content": "Give me endpoint for ServiceNow"}
])

result = app.invoke(initial_state)

print("\n\nFinal Result:\n")
print(result)
