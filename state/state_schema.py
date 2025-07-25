from typing import TypedDict, Optional, Dict, Any, List
import pandas as pd

class GraphState(TypedDict):
    df: pd.DataFrame
    goal: str
    target_column: str
    null_suggestions: Optional[Dict[str, str]]
    null_critique: Optional[Dict[str, str]]
    null_iterations: Optional[int]
    duplicate_suggestion: Optional[str]
    duplicate_critique: Optional[str]
    duplicate_iterations: Optional[int]
    type_suggestion: Optional[Dict[str, str]]
    type_suggestions: Optional[List[str]]
    type_analysis: Optional[Dict[str, Any]]
    type_iterations: Optional[int]