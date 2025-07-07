from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from langgraph.graph.message import add_messages
from pydantic import BaseModel,field_validator

class State(BaseModel):
    messages: Annotated[list,add]

