# File: langgraph_nodes/duplicate_check.py

def duplicate_check_node(state):
    df = state["df"]
    duplicate_rows = df[df.duplicated()].shape[0]
    suggestion = f"Found {duplicate_rows} duplicates. Consider dropping duplicates."
    state["duplicate_suggestion"] = suggestion
    
    # Update iteration counter
    state["duplicate_iterations"] = state.get("duplicate_iterations", 0) + 1
    
    return state