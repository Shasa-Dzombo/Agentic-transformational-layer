import pandas as pd
from config.llm_config import llm, NULL_STRATEGY_PROMPT

def null_check_node(state):
    df = state["df"]
    nulls = df.isnull().sum()
    total = len(df)
    
    # Basic null analysis
    null_analysis = {
        col: {
            "null_count": int(nulls[col]),
            "null_percentage": round(nulls[col]/total*100, 2)
        }
        for col in df.columns if nulls[col] > 0
    }
    
    # Get LLM suggestions if nulls exist
    if null_analysis:
        # Prepare context for LLM (removed target_column)
        context = {
            "goal": state.get("goal", ""),
            "null_analysis": str(null_analysis)
        }
        
        # Get LLM response
        prompt = NULL_STRATEGY_PROMPT.format(**context)
        response = llm.invoke(prompt)
        
        state["null_suggestions"] = response.content
        state["null_analysis"] = null_analysis
    else:
        state["null_suggestions"] = "No null values found."
        state["null_analysis"] = {}
    
    # Update iteration counter
    state["null_iterations"] = state.get("null_iterations", 0) + 1
    
    return state