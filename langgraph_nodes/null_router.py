def null_router_node(state):
    """Router function to determine next step based on null values"""
    critique = state.get("null_critique", "no")  # Now expects string, not dict
    
    # Check iteration counter to prevent infinite loops
    null_iterations = state.get("null_iterations", 0)
    
    # Check if null handling is needed based on LLM critique
    if critique == "yes" and null_iterations < 2:  # Reduced limit
        return "null_check"  # Loop back for more processing
    else:
        return "duplicate_check"  # Move to next stage