# File: langgraph_nodes/drop_critic.py

def drop_critic_node(state):
    state["drop_critique"] = "yes"
    return state
