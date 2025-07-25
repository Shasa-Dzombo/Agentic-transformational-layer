def null_critic_node(state):
    suggestions = state.get("null_suggestions", {})
    critique = {
        col: ("yes" if "Consider" in suggestion else "no")
        for col, suggestion in suggestions.items()
    }
    state["null_critique"] = critique
    return state