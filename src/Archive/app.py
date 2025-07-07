from langgraph.graph import StateGraph,END
from utils.statev2 import State
from utils.nodev2 import chatbot,node2

graph = StateGraph(State)
graph.add_node("chatbot", chatbot)  # Assuming tools is another function similar to chatbo
graph.set_entry_point("chatbot")
graph.add_node("node2", node2)
graph.add_edge("chatbot", "node2")
#graph.set_finish_point("chatbot")
graph.add_edge("node2", END)  # Loop back to chatbot for continuous interaction
app = graph.compile()

state = {
    "messages": [{"role":"system",
                 "content":"You are expert in Indian politics and current affairs."},
                 {"role":"user",
                 "content":"Give me endpoint for ServiceNow"},
    ]
}


response = app.invoke(state)

#print(response)