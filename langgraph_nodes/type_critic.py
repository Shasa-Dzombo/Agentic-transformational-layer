from typing import Dict, Any
import pandas as pd
from config.llm_config import llm, TYPE_STRATEGY_PROMPT

def type_critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes and critiques data types using LLM suggestions.
    """
    df = state.get("df")
    if df is None:
        return state
    
    # Get the type suggestions from type_check_node
    type_suggestion = state.get("type_suggestion", "")
    type_analysis = state.get("type_analysis", {})
    
    # Use LLM to critique the type analysis
    if type_analysis:
        context = {
            "goal": state.get("goal", ""),
            "type_analysis": str(type_analysis),
            "llm_suggestions": type_suggestion
        }
        
        critique_prompt = f"""
        You are a data type critic. Evaluate the following type analysis and LLM suggestions.
        
        User Goal: {context['goal']}
        Type Analysis: {context['type_analysis']}
        LLM Suggestions: {context['llm_suggestions']}
        
        Determine if immediate type optimization is needed for preprocessing. 
        Respond with 'yes' if types need optimization, 'no' if current types are acceptable.
        Consider the user's preprocessing goal and data quality requirements.
        """
        
        response = llm.invoke(critique_prompt)
        critique_result = response.content.strip().lower()
        
        # Extract yes/no decision
        if "yes" in critique_result:
            state["type_critique"] = "yes"
        else:
            state["type_critique"] = "no"
            
        # Store the full critique reasoning
        state["type_critique_reasoning"] = response.content
    else:
        state["type_critique"] = "no"
        state["type_critique_reasoning"] = "No type analysis available"
    
    return state