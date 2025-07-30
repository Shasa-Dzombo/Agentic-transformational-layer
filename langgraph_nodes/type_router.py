# File: langgraph_nodes/type_router.py

def type_router_node(state):
    """Router function to determine next step after type analysis"""
    critique = state.get("type_critique", "no")
    
    # Check iteration counter to prevent infinite loops
    type_iterations = state.get("type_iterations", 0)
    
    # Check if type optimization is needed based on LLM critique
    if critique == "yes" and type_iterations < 2:  # Reduced limit
        return "feature_engineer"
    else:
        return "drop_check"  # Always move to end eventually