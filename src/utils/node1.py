from langchain_openai import AzureChatOpenAI
import os
from utils.state import State
from langchain_core.messages import HumanMessage,AIMessage

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


Message=AIMessage("AI Message: sss")
# llm = AzureChatOpenAI(
#         azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), 
#         api_key=os.getenv("AZURE_OPENAI_KEY"),
#         api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#         azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

#         )
class llm:
    @staticmethod
    def invoke(messages):
        return Message
        #return "This is a mock response for: " + str(messages)
    


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}