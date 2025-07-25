from typing import Dict, Any
import pandas as pd

def type_critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes and critiques data types in the DataFrame.
    Suggests type conversions and identifies potential issues.
    """
    df = state.get("df")
    if df is None:
        return state
    
    # Analyze data types
    type_analysis = {}
    suggestions = []
    
    for column in df.columns:
        col_data = df[column]
        current_dtype = str(col_data.dtype)
        
        # Check for potential type improvements
        if current_dtype == 'object':
            # Check if it can be converted to numeric
            try:
                pd.to_numeric(col_data, errors='raise')
                suggestions.append(f"Column '{column}' can be converted to numeric")
            except (ValueError, TypeError):
                # Check if it's categorical
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    suggestions.append(f"Column '{column}' should be categorical")
        
        # Check for missing values that affect type inference
        null_count = col_data.isnull().sum()
        if null_count > 0:
            suggestions.append(f"Column '{column}' has {null_count} missing values")
        
        type_analysis[column] = {
            'current_type': current_dtype,
            'null_count': int(null_count),
            'unique_count': int(col_data.nunique())
        }
    
    # Update state with type analysis
    state['type_analysis'] = type_analysis
    state['type_suggestions'] = suggestions
    
    return state