# File: langgraph_nodes/duplicate_critic.py

def duplicate_critic_node(state):
    suggestion = state.get("duplicate_suggestion", "")
    state["duplicate_critique"] = "yes" if "Consider" in suggestion else "no"
    return state
