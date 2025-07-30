# File: langgraph_nodes/duplicate_critic.py
from config.llm_config import llm

def duplicate_critic_node(state):
    """
    Critiques duplicate analysis using LLM suggestions.
    """
    duplicate_suggestion = state.get("duplicate_suggestion", "")
    duplicate_analysis = state.get("duplicate_analysis", {})
    
    if duplicate_suggestion and "No duplicate" not in duplicate_suggestion:
        context = {
            "goal": state.get("goal", ""),
            "duplicate_suggestion": duplicate_suggestion,
            "duplicate_analysis": str(duplicate_analysis)
        }
        
        critique_prompt = f"""
        You are a duplicate data critic. Evaluate the duplicate analysis and recommendations.
        
        User Goal: {context['goal']}
        Duplicate Analysis: {context['duplicate_analysis']}
        LLM Suggestions: {context['duplicate_suggestion']}
        
        Determine if immediate action is needed for duplicates.
        Consider the impact on the user's preprocessing goal and data quality.
        Respond with 'yes' if duplicates need immediate handling, 'no' if they can be ignored.
        """
        
        response = llm.invoke(critique_prompt)
        critique_result = response.content.strip().lower()
        
        # Extract yes/no decision
        if "yes" in critique_result:
            state["duplicate_critique"] = "yes"
        else:
            state["duplicate_critique"] = "no"
            
        # Store the full critique reasoning
        state["duplicate_critique_reasoning"] = response.content
    else:
        state["duplicate_critique"] = "no"
        state["duplicate_critique_reasoning"] = "No duplicates found"
    
    return state
