def null_router_node(state):
    """Router function to determine next step based on null values"""
    critique = state.get("null_critique", {})
    
    # Check if any column needs null handling
    needs_handling = any(val == "yes" for val in critique.values())
    
    # Check iteration counter to prevent infinite loops
    null_iterations = state.get("null_iterations", 0)
    
    if needs_handling and null_iterations < 2:  # Reduced limit
        return "null_check"  # Loop back for more processing
    else:
        return "duplicate_check"  # Move to next stage