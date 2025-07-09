import aiohttp
import asyncio
import dotenv,os,json,requests
from langchain.tools import tool

dotenv.load_dotenv()

usrr=os.getenv("usrr")
pwdd=os.getenv("pwdd")

@tool(return_direct=True)
def GetsAnyDetail(filter: str):
    """
    TO use this get call createAPIParam tool first. 
    Gets custom details from ServiceNow based on the filter created by createAPIParam tool.

    Args:
        filter: URL parameter filter to get details from API
    """
    print("filter ------",filter)
    # Define headers and URL
    headers = {"Accept": "application/json"}
    api_url = f"https://dev185303.service-now.com/api/now/table{filter}"

    # Basic Authentication
    auth = (usrr, pwdd)

    # Make the GET request
    response = requests.get(api_url,auth=auth, headers=headers, verify=False)
    print("response from getanydetail =====",response.json())
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return "Not found."


def CreateANewRecord_in_SNOW(data: str):
    """Creates a new record in ServiceNow in required table as per the parameter provided by createEndPointPayload tool.
    
    Args:
        data: contains table name and payload in the format "table_name|payload"    
    """
    # Split the data into table name and payload
    parts = data.split("|")
    table_name = parts[0].strip()
    payload = parts[1].strip()
    payload_dict = json.loads(payload)

    # Define API URL and headers
    headers = {"Accept": "application/json"}
    api_url = f"https://dev185303.service-now.com/api/now/table/{table_name}"

    # Basic Authentication
    auth = usrr, pwdd

    # Make the POST request
    response = requests.post(api_url, headers=headers, auth=auth, json=payload_dict, verify=False)

    if response.status_code == 201:
        result = response.json()
        return result["result"]
    else:
        return (
            "Record creation failed. "
            + str(response.status_code)
            + " "
            + str(response.reason)
            + " "
            + response.text
        )