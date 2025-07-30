# File: langgraph_nodes/type_check.py

import pandas as pd
from config.llm_config import llm, TYPE_STRATEGY_PROMPT

def type_check_node(state):
    df = state["df"]
    
    # Analyze data types
    type_analysis = {}
    for column in df.columns:
        col_data = df[column]
        type_analysis[column] = {
            'current_type': str(col_data.dtype),
            'null_count': int(col_data.isnull().sum()),
            'unique_count': int(col_data.nunique()),
            'memory_usage': col_data.memory_usage(deep=True)
        }
    
    # Get LLM suggestions (removed target_column)
    context = {
        "goal": state.get("goal", ""),
        "type_analysis": str(type_analysis)
    }
    
    prompt = TYPE_STRATEGY_PROMPT.format(**context)
    response = llm.invoke(prompt)
    
    state["type_suggestion"] = response.content
    state["type_analysis"] = type_analysis
    
    # Update iteration counter
    state["type_iterations"] = state.get("type_iterations", 0) + 1
    
    return state
