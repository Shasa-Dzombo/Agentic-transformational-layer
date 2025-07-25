# File: langgraph_nodes/type_check.py

def type_check_node(state):
    df = state["df"]
    dtypes = df.dtypes.apply(lambda dt: str(dt))
    state["type_suggestion"] = dtypes.to_dict()
    
    # Update iteration counter
    state["type_iterations"] = state.get("type_iterations", 0) + 1
    
    return state
