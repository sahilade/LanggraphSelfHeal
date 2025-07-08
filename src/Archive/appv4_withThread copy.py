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
from langgraph.graph import StateGraph, START,END
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda
from utils.tools import GetAPIEndpoint, get_weather, get_wind
from utils.tools_thread import GetsAnyDetail,CreateANewRecord_in_SNOW_async


load_dotenv(override=True)



class State(BaseModel):
    messages: List[BaseMessage] = []
    previous_messages: Optional[List[BaseMessage]] = None


#--------------------



def createAPIParam(Query: str):
    """This tool is ONLY used for any kind of GET Request.
    Gives ServiceNow TableName and Create API parameter based on the query from user. Need to be used before getting details from other tools.
    Args: Query - Natural language query
    
    """
  
    FilterPrompt=f""" 
        You are ServiceNow API Expert. 
        As per the requirement you need to create API parameters to get the required detail.
        You will be provided with user query and you need to create sysparm_query on the required table with
        param filter.

         Follow this thought process:
        1. Identify the service now table name based on the query from your knowledge and below examples.
        2. Understand if this service now table would need sys_id for filtering or not.
        3. If sys_id is required, you extract sys_id first by getting required sys_id for respective incident, RITM etc
        4. Then, Create a sysparm_query based on sys_id or other fields. 
        5. If the query is not clear, ask the user for more information.

        strict output format 
        /*ServiceNow_table_name*?sysparm_query=*sysparm_query*&sysparm_fields=*fields*&sysparm_limit=*limit*'
        
        Examples:
        example 1 : Query - Give me incidents for user abel tuter. Answer: /incident?sysparm_query=sys_created_by=Abel.tuter
        example 2: Query - Give me active incidents. Answer: /incident?sysparm_query=stateIN1,2
        example 3: Query - Give me active incidents of user abc def. Answer : /incident?sysparm_query=sys_created_by=abc.def^stateIN1,2&sysparm_fields=number,state,sys_id,sys_created_by
        example 4: Query - Give me incidents for user abel tuter and active incidents. Answer: /incident?sysparm_query=sys_created_by=Abel.tuter^stateIN1,2
        example 5: Query -  Give me details of users with name "andrew" /sys_user?sysparm_query=user_nameLIKEandrew&sysparm_fields=sys_id,user_name,email
        example 6: Query - Give me users details of user Andrew Och. Answer: /sys_user?sysparm_query=user_name=andrew.och&sysparm_fields=sys_id,user_name,email
        example 7: Query - Give me work notes for incident INC12321. Answer: /sys_journal_field?sysparm_query=element_id=f12ca184735123002728660c4cf6a7ef&element=work_notes
        example 8: Query - Give me additional comments for incident INC12321. Answer: /sys_journal_field?sysparm_query=element_id=f12ca184735123002728660c4cf6a7ef&element=comments
       
        Answer below query based on above example. 
        Filter should preceed with "/"
        
        Only if you are not able to create the filter, then refer below link
        https://www.servicenow.com/docs/bundle/yokohama-api-reference/page/integrate/inbound-rest/concept/c_RESTAPI.html
        Query - {Query}. Answer - ?

    """

    #print(FilterPrompt)
    rslt= llm_client.invoke(FilterPrompt)
    #print(rslt)
    return rslt



def createEndPointPayload(Query: str):
    """This tool is NOT used for any kind of GET Request.
    Create endpoint and payload for any Create/Insert/Post/PATch/PUT/DElete request apart from Incident and RITM (for which separate tools are created).
    Args: Query - Natural language query
    
    """
    FilterPrompt=f""" 
        You are ServiceNow API explorer. 
        As per the requirement you need to give ServiceNow Tablename and payload to get the required detail.
        You will be provided with user query.

        Follow this thought process:
        1. Identify the service now table name based on the query from your knowledge and below examples.
        2. Understand if this service now table would need sys_id for filtering or not.
        3. If sys_id is required, you extract sys_id first by getting required sys_id for respective incident, RITM etc
        4. Then, Create a payload based on the query
        5. If the query is not clear, ask the user for more information.

        Base url is already there. You need to give tablename and create payload for the request.

        Return 2 fields in below in JSON string format for strict parsing. Do not return anything else or any keyword
        {{
            table_name = /ServiceNow table_name for that request like /incident, /sys_user, /sc_req_item etc.
            Payload =  {{"field1": "value1", "field2": "value2"}}
        }}
        
        examples:
        example 1: Query - Post additional comments for incident INC12321. Answer: table_name=/sys_journal_field, Payload = {{"element_id": "f12ca184735123002728660c4cf6a7ef", "element": "comments", "value": "This is a test comment","name":"incident"}}
        Note : for posting additional comments or work_notes you will need to fetch sys_id of incident/RITM first using createAPIParam tool.
       
        Answer below query based on above example. 
        table_name should always preceed with "/"
        
        Only if you are not able to create the filter, then refer below link
        https://www.servicenow.com/docs/bundle/yokohama-api-reference/page/integrate/inbound-rest/concept/c_RESTAPI.html
        Query - {Query}. Answer - ?

    """

    #print(FilterPrompt)
    rslt= llm_client.invoke(FilterPrompt)
    print("*******************",rslt.content)
    #return rslt.content["table_name"]+"|"+json.dumps(rslt.content["Payload"])  # Return table name and payload as a string separated by '|'
    print("parsing result")
    print(type(rslt.content))
    parsed = json.loads(rslt.content)
    print("parsed result")
    table_name = parsed["table_name"]
    print("parsed table name")
    payload = parsed["Payload"]
    print("Parsed Payload")
    return table_name + "|" + str(json.dumps(payload))


#>>>>>>>>>>>>>>>>>>>>

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
    #print(messages)
    for m in messages:
        print("\n Looping on messages : ", m)
        if isinstance(m, HumanMessage) or (isinstance(m,BaseMessage) and m.type=="human"):
            role = "user"
            content = m.content
        elif isinstance(m, AIMessage)  or (isinstance(m,BaseMessage) and m.type=="ai"):
            role = "assistant"
            content = m.content
        elif isinstance(m, SystemMessage)  or (isinstance(m,BaseMessage) and m.type=="system"):
            role = "system"
            content = m.content
        elif isinstance(m, ToolMessage) or (isinstance(m,BaseMessage) and m.type=="tool"):
            role = "tool_response"
            content = json.dumps({
                    "tool_call_id": m.tool_call_id,
                    "name": m.name,
                    "content": m.content
                })
        else:
            raise ValueError(f"Unsupported message type: {type(m)}")
        result.append({"role": role, "content": content})
    #print(result)
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
        #print( response.json())
        # Build an AIMessage (content and tool_calls both supported)
        
        # return AIMessage(
        #     content=choice.get("content", ""),
        #     additional_kwargs={"tool_calls": choice.get("tool_calls", [])}
        # )

        content = choice.get("content") or ""
        #print(f"[DEBUG] Content: {content}")
        tool_calls = choice.get("tool_calls")
        #print(f"[DEBUG] Tool calls: {tool_calls}")
        print("AIMSG===============================")
        AIMsg=AIMessage(
            content=content,
            additional_kwargs={"tool_calls": tool_calls} if tool_calls else {}
        )
        #print(AIMsg)
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
    #print("^"*100, "\n[DEBUG] AGENT NODE")
    #print(state.messages)
    response = internal_llm.invoke(state.messages, functions=tools_schemas )
    return State(messages=state.messages + [response], previous_messages=state.messages)

# =====================================================
# User Input Node
# def user_input_node(state: State) -> State:
#     print("\n========== Conversation So Far ==========")
#     for m in state.messages:
#         if isinstance(m, SystemMessage):
#             continue
#         if isinstance(m, HumanMessage):
#             print(f">> User: {m.content}")
#         elif isinstance(m, AIMessage):
#             print(f">> Assistant: {m}")
#             if m.content:
#                 print(f"**Assistant: {m.content}")
#             if "tool_calls" in m.additional_kwargs:
#                 for call in m.additional_kwargs["tool_calls"]:
#                     name = call["function"]["name"]
#                     args = call["function"]["arguments"]
#                     print(f">> Assistant (function call): {name}({args})")
#     print("=========================================")

#     user_text = input("\nYour message (or 'exit'): ").strip()
#     if user_text.lower() == "exit":
#         print("\n[Conversation ended by user]")
#         exit(0)

#     return State(messages=state.messages + [HumanMessage(content=user_text)])

def tool_output_node(state: State) -> State:
    print("^^"*50, "\n[DEBUG] TOOL OUTPUT NODE\n")
    #print(state,"\n")
    return State(messages=state.previous_messages + state.messages)

# =====================================================

tool_node = ToolNode(tools=tools)

def tool_condition(state: State) -> str:
    print("^^"*100, "\n[DEBUG] Checking tool condition")
    last = state.messages[-1]
    #print(f"[DEBUG] Last message: {last}")
    if isinstance(last, AIMessage) and last.additional_kwargs.get("tool_calls"):
        print("tools")
        return "tools"
    print("end")
    return "END"

# =====================================================
# Buiild LangGraph
graph = StateGraph(State)
graph.add_node("agent", RunnableLambda(agent_node))
graph.add_node("tools", tool_node)
#graph.add_node("user_input", RunnableLambda(user_input_node))
graph.add_node("tool_output", RunnableLambda(tool_output_node))

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    tool_condition
)
graph.add_edge("tools", "tool_output")
graph.add_edge("tool_output", "agent")
graph.add_edge("agent", END)

app = graph.compile()

# =====================================================

initial_state = State(messages=[
    SystemMessage(content="You are helpful assistant."),
    HumanMessage(content="give me endpoint of SAP")
])

SystemMsg=SystemMessage(content="You are helpful assistant.")

def invoke(text:str,context: Optional[List] = None):
    print("Text -",text)
    #if context is not None:
        # print("context - ",context)
        # print("Context main message :",context["messages"])
        # print("Context Prev message :",context["previous_messages"])
        # #return "Hello from AI"

    if context is None:
        context = []
        l= app.invoke(State(messages=[SystemMsg,HumanMessage(content=text)]))
    else:
        l= app.invoke(State(messages=context["messages"]+[HumanMessage(content=text)],previous_messages=context["previous_messages"]))
    #print(l["messages"][-1].content)
    return l