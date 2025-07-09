import requests
import os,dotenv
import json
dotenv.load_dotenv(override=True)
import time
import httpx

#LLM Object
class InternalLLM:
    def __init__(self, api_url, team_id, project_id, api_key, model_name="gpt-4o-mini", temperature=0.5):
        self.api_url = api_url
        self.team_id = team_id
        self.project_id = project_id
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
    
    def invoke(self,message:str):
        payload = {
            "prompt": message,
            "generation_model": self.model_name,
            "max_tokens": 1000,
            "temperature": self.temperature,
            "n": 1,
            "raw_response": False,
            "stream":True

        }

      
        headers = {
            "Authorization": f"Bearer {os.getenv('okta_token')}",
            "team_id": self.team_id,
            "project_id": self.project_id,
            "x-pepgenx-apikey":self.api_key,
            "Content-Type": "application/json"
        }

        # Make the request
        print("calling llm") 
        # #response = requests.post(self.api_url, headers=headers, json=payload,stream=True)
        # response = requests.post(self.api_url, headers=headers, json=payload,stream=True)
        # print("received llm response")
        # for chunk in response.iter_lines( chunk_size=1):
        #     if chunk:
        #         line = chunk.decode('utf-8').strip()
        #         if line.startswith('data:'):
        #             data_str = line[len('data:'):].strip()
        #             try:
        #                 data_json = json.loads(data_str)
        #                 text_piece = data_json.get('response')
        #                 if text_piece is not None:
        #                     print(text_piece, end='', flush=True)
        #             except json.JSONDecodeError:
        #                 print(f"\n[WARN] Could not parse JSON: {data_str}")
        
        with httpx.stream("POST", self.api_url, headers=headers, json=payload,verify=False) as r:
            print("received llm response")
            for chunk in r.iter_lines():
                if chunk:
                    line = chunk.strip()
                    if line.startswith('data:'):
                        data_str = line[len('data:'):].strip()
                        try:
                            data_json = json.loads(data_str)
                            text_piece = data_json.get('response')
                            if text_piece is not None:
                                print(text_piece, end='', flush=True)
                                time.sleep(0.08)
                        except json.JSONDecodeError:
                            print(f"\n[WARN] Could not parse JSON: {data_str}")

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


# Define the API endpoint
#url = 'https://apim-na.qa.mypepsico.com/cgf/pepgenx/v2/llm/openai/generate-response'

# Make the GET request
internal_llm.invoke("give me 100 words essay on atlantis lost city")

# # Check if request was successful
# if response.status_code == 200:
#     # Parse and print JSON
#     data = response.json()
#     print("Response in JSON:")
#     print(data)
# else:
#     print(f"Request failed with status code: {response.json()}")
