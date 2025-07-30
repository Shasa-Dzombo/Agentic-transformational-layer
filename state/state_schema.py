from typing import TypedDict, Optional, Dict, Any, List
import pandas as pd

class GraphState(TypedDict):
    df: pd.DataFrame
    goal: str
    target_column: str
    
    # Null handling
    null_suggestions: Optional[str]
    null_critique: Optional[str]
    null_iterations: Optional[int]
    
    # Duplicate handling
    duplicate_suggestion: Optional[str]
    duplicate_analysis: Optional[Dict[str, Any]]
    duplicate_critique: Optional[str]
    duplicate_critique_reasoning: Optional[str]
    duplicate_iterations: Optional[int]
    
    # Type handling
    type_suggestion: Optional[str]
    type_analysis: Optional[Dict[str, Any]]
    type_critique: Optional[str]
    type_critique_reasoning: Optional[str]
    type_iterations: Optional[int]