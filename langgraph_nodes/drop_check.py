# File: langgraph_nodes/drop_check.py

def drop_check_node(state):
    df = state["df"]
    suggestions = [col for col in df.columns if df[col].nunique() == 1]
    state["drop_suggestion"] = suggestions
    return state
