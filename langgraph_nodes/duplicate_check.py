# File: langgraph_nodes/duplicate_check.py

import pandas as pd
from config.llm_config import llm, DUPLICATE_STRATEGY_PROMPT

def duplicate_check_node(state):
    df = state["df"]
    
    # Analyze duplicates
    duplicate_mask = df.duplicated()
    duplicate_count = duplicate_mask.sum()
    
    if duplicate_count > 0:
        duplicate_analysis = {
            "total_duplicates": int(duplicate_count),
            "percentage": round(duplicate_count/len(df)*100, 2),
            "duplicate_rows": df[duplicate_mask].to_dict('records')[:3]  # First 3 examples
        }
        
        # Get LLM suggestions (removed target_column)
        context = {
            "goal": state.get("goal", ""),
            "duplicate_analysis": str(duplicate_analysis)
        }
        
        prompt = DUPLICATE_STRATEGY_PROMPT.format(**context)
        response = llm.invoke(prompt)
        
        state["duplicate_suggestion"] = response.content
        state["duplicate_analysis"] = duplicate_analysis  # Store for critic
    else:
        state["duplicate_suggestion"] = "No duplicate rows found."
        state["duplicate_analysis"] = {}
    
    # Update iteration counter
    state["duplicate_iterations"] = state.get("duplicate_iterations", 0) + 1
    
    return state