def null_check_node(state):
    df = state["df"]
    nulls = df.isnull().sum()
    total = len(df)
    suggestions = {
        col: f"{round(nulls[col]/total*100, 2)}% nulls. Consider dropping or imputing." 
        for col in df.columns if nulls[col] > 0
    }
    state["null_suggestions"] = suggestions
    
    # Update iteration counter
    state["null_iterations"] = state.get("null_iterations", 0) + 1
    
    return state