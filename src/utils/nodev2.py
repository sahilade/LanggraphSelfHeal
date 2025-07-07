import os,json
from utils.statev2 import State
import requests 
from dotenv import load_dotenv, find_dotenv
from utils.tools import GetAPIEndpoint


state = {
    "messages": [{"role":"system",
                 "content":"You are expert in Indian politics and current affairs."},
                 {"role":"user",
                 "content":"who became prime minister after Rajendra Prasant ?"},
    ]
}


tool=[  {"type":"function",
         "function":   
            {
                "name":"GetAPIEndpoint",
                "description":"Get API Endpoint based on the job type.",
                "parameters":{
                    "type":"object",
                    "properties":{
                        "x":{
                            "type":"string",
                            "description":"Job Type. Can be sap, salesforece or any other."
                    }},
                    "required":["x"],
                    "additionalProperties": False 
                },
                "strict":True
            }}
        
        ]




def PayloadCreater(messages):
    return json.dumps({
  "prompt": json.dumps(messages),
  "generation_model": "gpt-4o-mini",
  "max_tokens": 100,
  "temperature": 0.5,
  "n": 1,
  "raw_response": True,
  "tools": tool,
  "tool_choice": "auto"
})

load_dotenv(find_dotenv(),verbose=True,override=True)
#print(os.getenv("okta_token"))

class llm:
    @staticmethod
    def invoke(messages):
        response=requests.post(
            url=os.getenv("api_url"),
            headers={
                "Authorization": f"Bearer {os.getenv('okta_token')}",
                "team_id": os.getenv("team_id"),
                "project_id": os.getenv("project_id"), 
                "x-pepgenx-apikey": os.getenv("apikey"),
                "Content-Type": "application/json"
            },
            data=PayloadCreater(messages)
        )
        if response.status_code == 200:
            print("Response received from LLM:", response.json())
            return {"role":response.json()["choices"][0]["message"]["role"],"content":response.json()["choices"][0]["message"]["content"]}
        else:
            return f"Error: {response.status_code} - {response.text}"
    


def chatbot(state: State):
    return {"messages": [llm.invoke(state.messages)]}

def node2(state: State):
    print("Node2 invoked with state:")

