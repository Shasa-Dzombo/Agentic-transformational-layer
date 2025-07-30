import pandas as pd
import os
from dotenv import load_dotenv
from langgraph_nodes.graph_builder import builder

# Load environment variables
load_dotenv()

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load entire CSV dataset without sampling
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns from {file_path}")
        print(f"Columns: {list(df.columns)}")
        print(f"Dataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return df
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

if __name__ == "__main__":
    # Configuration - Update these for your dataset
    CSV_FILE_PATH = "C:/Users/Amoro/APHRC_extractor/data_analyst_agent/data/df_sample.csv"
    ANALYSIS_GOAL = "do the necessary pre_checks and preprocessing of the dataset"

    # Load the entire dataset
    print("Loading dataset...")
    df = load_csv_data(CSV_FILE_PATH)
    
    if df is None:
        print("Failed to load data. Please check the file path and try again.")
        exit(1)
    
    # Display dataset overview
    print("\n" + "="*50)
    print("DATASET OVERVIEW:")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    
    # Prepare state for the workflow (no target column needed)
    state = {
        "df": df,
        "goal": ANALYSIS_GOAL
    }
    
    print(f"\n" + "="*50)
    print("STARTING DATA PREPROCESSING WORKFLOW")
    print("="*50)
    print(f"Goal: {ANALYSIS_GOAL}")
    print(f"Processing {len(df)} rows...")
    
    # Run the complete preprocessing workflow
    app = builder.compile()
    final_state = app.invoke(state)
    
    print("\n" + "="*60)
    print("PREPROCESSING RESULTS:")
    print("="*60)
    
    # Display results in organized way
    for key, value in final_state.items():
        if key == "df":
            processed_df = value
            print(f"Processed DataFrame: {processed_df.shape}")
        elif "reasoning" in key:
            print(f"\n{key.replace('_', ' ').title()}:")
            print("-" * 40)
            print(f"{value}")
        elif "suggestions" in key:
            print(f"\n{key.replace('_', ' ').title()}:")
            print("-" * 40)
            print(f"{value}")
        else:
            print(f"{key}: {value}")
    
    # Optional: Save processed dataset
    print(f"\n" + "="*50)
    save_option = input("Save processed dataset to CSV? (y/n): ").lower().strip()
    if save_option == 'y':
        output_path = "data/processed_dataset.csv"
        os.makedirs("data", exist_ok=True)  # Create data directory if it doesn't exist
        final_state["df"].to_csv(output_path, index=False)
        print(f"✅ Processed dataset saved to {output_path}")
        print(f"   Original shape: {df.shape}")
        print(f"   Processed shape: {final_state['df'].shape}")
    
    print("\n🎉 Data preprocessing workflow completed successfully!")
