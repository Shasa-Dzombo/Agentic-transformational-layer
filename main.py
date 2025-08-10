import pandas as pd
import os
import json
from dotenv import load_dotenv
from langgraph_nodes.graph_builder import builder
from config.enhanced_ai_schema_mapper import EnhancedAISchemaMapper  # Enhanced AI mapper
from database.db_handler import SupabaseClientHandler

# Load environment variables
load_dotenv()

def load_csv_data(file_path: str) -> pd.DataFrame:
    """Load entire CSV dataset without sampling"""
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

def setup_supabase_client():
    """Setup Supabase client from environment variables"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_CLIENT_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("""
        Supabase credentials not found. Please set in your .env file:
        
        SUPABASE_URL=https://your-project-ref.supabase.co
        SUPABASE_CLIENT_KEY=your_supabase_service_role_key
        """)
    
    return supabase_url, supabase_key

if __name__ == "__main__":
    # Configuration
    CSV_FILE_PATH = "C:/Users/Amoro/APHRC_extractor/data_analyst_agent/data/df_sample.csv"
    SCHEMA_FILE_PATH = "C:/Users/Amoro/APHRC_extractor/data_analyst_agent/database/database_schema.json"
    ANALYSIS_GOAL = "do the necessary pre_checks and preprocessing of the dataset for multi-table database loading"

    # Setup Supabase client
    try:
        supabase_url, supabase_key = setup_supabase_client()
        print("✅ Supabase client configuration loaded")
        print(f"   URL: {supabase_url}")
        print(f"   Key: {supabase_key[:20]}...")
    except ValueError as e:
        print(f"❌ {e}")
        exit(1)

    # Step 1: Load and preprocess data
    print("="*80)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    df = load_csv_data(CSV_FILE_PATH)
    if df is None:
        print("Failed to load data. Please check the file path and try again.")
        exit(1)
    
    # Display dataset overview
    print("\n" + "="*50)
    print("DATASET OVERVIEW:")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"Sample columns: {list(df.columns[:10])}")
    
    # Prepare state for the workflow
    state = {
        "df": df,
        "goal": ANALYSIS_GOAL
    }
    
    print(f"\n" + "="*50)
    print("PREPROCESSING WORKFLOW")
    print("="*50)
    
    # Run the complete preprocessing workflow
    app = builder.compile()
    final_state = app.invoke(state)
    
    processed_df = final_state["df"]
    preprocessing_results = {
        "null_suggestions": final_state.get("null_suggestions", ""),
        "duplicate_suggestions": final_state.get("duplicate_suggestion", ""),
        "type_suggestions": final_state.get("type_suggestion", ""),
        "type_analysis": final_state.get("type_analysis", {})
    }
    
    print(f"✅ Preprocessing completed. Shape: {processed_df.shape}")
    
    # Step 2: AI-Powered Schema Mapping
    print("\n" + "="*80)
    print("STEP 2: AI-POWERED SCHEMA MAPPING")
    print("="*80)
    
    try:
        # Initialize Enhanced AI schema mapper
        ai_mapper = EnhancedAISchemaMapper(SCHEMA_FILE_PATH)
        
        # Show schema intelligence
        print(f"🧠 Schema intelligence built:")
        print(f"   Autonomous table analysis: {len(ai_mapper.schema_intelligence['table_purposes'])}")
        print(f"   Semantic field mapping: {sum(len(fields) for fields in ai_mapper.schema_intelligence['field_semantics'].values())}")
        
        # Use autonomous AI to map data
        mapped_tables = ai_mapper.map_dataframe_to_tables(processed_df, preprocessing_results)
        
        print(f"\n✅ Autonomous AI schema mapping completed")
        print(f"📋 Intelligently created tables: {list(mapped_tables.keys())}")
        
    except Exception as e:
        print(f"❌ Autonomous AI mapping failed: {e}")
        exit(1)
    
    # Step 3: Supabase client operations
    print("\n" + "="*80)
    print("STEP 3: SUPABASE CLIENT OPERATIONS")
    print("="*80)
    
    try:
        # Load schema for database operations
        with open(SCHEMA_FILE_PATH, 'r') as f:
            schema_config = json.load(f)
        
        # Initialize Supabase client handler
        db_handler = SupabaseClientHandler(supabase_url, supabase_key, schema_config)
        
        # Test connection
        if not db_handler.test_connection():
            print("⚠️  Connection test had issues but proceeding...")
        
        # Validate all tables
        validation_results = db_handler.validate_all_tables(mapped_tables)
        
        # Check if we can proceed with saving
        all_valid = all(result['valid'] for result in validation_results.values())
        
        print(f"\n📊 Validation Summary:")
        print(f"   Status: {'✅ All Valid' if all_valid else '⚠️  Some Issues'}")
        
        # Show validation details
        for table_name, result in validation_results.items():
            if result['errors']:
                print(f"\n❌ {table_name} Errors:")
                for error in result['errors']:
                    print(f"   • {error}")
            if result['warnings']:
                print(f"\n⚠️  {table_name} Warnings:")
                for warning in result['warnings']:
                    print(f"   • {warning}")
        
        # Save to Supabase
        proceed = all_valid or input(f"\nProceed with {len(mapped_tables)} tables despite warnings? (y/n): ").lower() == 'y'
        
        if proceed:
            save_results = db_handler.save_mapped_tables(mapped_tables)
            successful_saves = sum(save_results.values())
            
            if successful_saves > 0:
                print(f"\n🎉 AI-powered Supabase loading completed!")
                print(f"   Database: Supabase (Client API)")
                print(f"   Tables saved: {successful_saves}/{len(mapped_tables)}")
                
                # Show final stats
                total_records = sum(len(df) for df in mapped_tables.values())
                print(f"   Total records: {total_records}")
                
                # Query sample data to verify
                print(f"\n📊 Verifying data in Supabase:")
                for table_name in list(mapped_tables.keys())[:2]:  # Check first 2 tables
                    try:
                        count = db_handler.get_table_count(table_name)
                        sample_data = db_handler.query_table(table_name, limit=3)
                        print(f"   ✅ {table_name}: {count} records")
                        if not sample_data.empty:
                            print(f"      Sample columns: {list(sample_data.columns)}")
                    except Exception as e:
                        print(f"   ⚠️  {table_name}: Could not verify - {e}")
                
            else:
                print(f"\n❌ Failed to save any tables to Supabase")
        else:
            print("❌ Supabase loading cancelled")
            
    except Exception as e:
        print(f"❌ Supabase operations failed: {e}")
        print(f"Error details: {str(e)}")
        exit(1)
    
    # Step 4: Final Summary
    print("\n" + "="*80)
    print("AI-POWERED SUPABASE WORKFLOW COMPLETED! 🎉")
    print("="*80)
    print(f"📥 Original data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"🔄 Processed data: {processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
    print(f"🤖 AI-mapped tables: {len(mapped_tables)}")
    
    for table_name, table_df in mapped_tables.items():
        print(f"   📋 {table_name}: {len(table_df)} records")
    
    print(f"💾 Database: Supabase (via Client API)")
    print(f"🌐 URL: {supabase_url}")
    print("\nYour data has been intelligently mapped and loaded into Supabase! 🚀")
    print("\n🔗 Access your data at: https://app.supabase.com/project/[your-project]/editor")
