from config.llm_config import llm

def null_critic_node(state):
    """
    Critiques null value analysis using LLM suggestions.
    """
    null_suggestions = state.get("null_suggestions", "")
    null_analysis = state.get("null_analysis", {})
    
    if null_analysis and null_suggestions:
        context = {
            "goal": state.get("goal", ""),
            "null_suggestions": null_suggestions,
            "null_analysis": str(null_analysis)
        }
        
        critique_prompt = f"""
        You are a data quality critic specializing in missing value analysis.
        
        User Goal: {context['goal']}
        Null Analysis: {context['null_analysis']}
        LLM Suggestions: {context['null_suggestions']}
        
        Evaluate if immediate action is needed for missing values based on:
        1. The percentage of missing values per column
        2. The impact on the user's preprocessing goal
        3. The severity and recommendations in the LLM suggestions
        
        Respond with 'yes' if null values need immediate handling, 'no' if they can be left as-is.
        Consider data quality requirements for preprocessing.
        """
        
        response = llm.invoke(critique_prompt)
        critique_result = response.content.strip().lower()
        
        # Extract yes/no decision
        if "yes" in critique_result:
            state["null_critique"] = "yes"
        else:
            state["null_critique"] = "no"
            
        # Store the full critique reasoning
        state["null_critique_reasoning"] = response.content
        
    else:
        state["null_critique"] = "no"
        state["null_critique_reasoning"] = "No null values found or no analysis available"
    
    return state