# File: langgraph_nodes/duplicate_router.py

def duplicate_router_node(state):
    """Router function to determine next step based on duplicate analysis"""
    critique = state.get("duplicate_critique", "no")
    
    # Check iteration counter to prevent infinite loops
    duplicate_iterations = state.get("duplicate_iterations", 0)
    
    # Check if duplicate handling is needed based on LLM critique
    if critique == "yes" and duplicate_iterations < 2:  # Reduced limit
        return "duplicate_check"  # Loop back for more processing
    else:
        return "type_check"  # Move to next stage
