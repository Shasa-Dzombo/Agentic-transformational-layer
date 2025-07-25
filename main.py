import pandas as pd
from langgraph_nodes.graph_builder import builder

if __name__ == "__main__":
    # Sample DataFrame
    df = pd.DataFrame({
        "age": [25, 30, 30, None, 22, None],
        "income": [50000, 60000, 60000, 55000, None, 60000],
        "gender": ["M", "F", "F", "M", "M", "M"]
    })

    # Sample state
    state = {
        "df": df,
        "goal": "Predict income",
        "target_column": "income"
    }

    app = builder.compile()
    final_state = app.invoke(state)

    print("\nFinal Processed State:")
    for key, value in final_state.items():
        print(f"{key}: {value}")
