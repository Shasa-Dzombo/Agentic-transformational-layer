# File: langgraph_nodes/type_router.py

def type_router_node(state):
    """Router function to determine next step after type analysis"""
    suggestions = state.get("type_suggestions", [])
    
    # Check iteration counter to prevent infinite loops
    type_iterations = state.get("type_iterations", 0)
    
    if suggestions and type_iterations < 2:  # Reduced limit
        return "feature_engineer"
    else:
        return "drop_check"  # Always move to end eventually