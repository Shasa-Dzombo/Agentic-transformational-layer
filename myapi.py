from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import os
import shutil
from main import load_csv_data, apply_smart_filter, setup_supabase_client
from config.enhanced_ai_schema_mapper import EnhancedAISchemaMapper
from database.db_handler import SupabaseClientHandler
from langgraph_nodes.graph_builder import builder
import json

@app.post("/process")
async def process_csv(
    file: UploadFile = File(...),
    use_ai_filtering: bool = Form(True)
):
    """
    Upload a CSV and run the full pipeline: smart filter, preprocess, schema map, Supabase save.
    Returns summary and table info.
    """
    # Save uploaded file to a temp location
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Load CSV
        raw_df = load_csv_data(temp_path)
        if raw_df is None:
            raise HTTPException(status_code=400, detail="Failed to load CSV data.")

        # Smart filter
        filtered_df = apply_smart_filter(raw_df, use_ai=use_ai_filtering)

        # Preprocessing workflow
        initial_state = {"df": filtered_df}
        app_graph = builder.compile()
        final_state = app_graph.invoke(initial_state)
        processed_df = final_state["df"]

        # Schema mapping
        SCHEMA_FILE_PATH = os.path.join("database", "database_schema.json")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
        mapper = EnhancedAISchemaMapper(SCHEMA_FILE_PATH, api_key)
        preprocessing_results = {
            "null_suggestions": "Preprocessing completed",
            "duplicate_suggestions": "No duplicates found",
            "type_suggestions": "Types optimized",
            "type_analysis": {}
        }
        mapped_tables = mapper.map_dataframe_to_tables(processed_df, preprocessing_results)

        # Supabase
        supabase_url, supabase_key = setup_supabase_client()
        with open(SCHEMA_FILE_PATH, 'r') as f:
            schema_config = json.load(f)
        db_handler = SupabaseClientHandler(supabase_url, supabase_key, schema_config)
        if not db_handler.test_connection():
            raise HTTPException(status_code=500, detail="Supabase connection failed.")
        validation_results = db_handler.validate_all_tables(mapped_tables)
        save_results = db_handler.save_mapped_tables(mapped_tables)

        # Clean up temp file
        os.remove(temp_path)

        # Prepare response
        return JSONResponse({
            "original_shape": raw_df.shape,
            "filtered_shape": filtered_df.shape,
            "processed_shape": processed_df.shape,
            "tables": {k: {"rows": len(v), "columns": len(v.columns)} for k, v in mapped_tables.items()},
            "validation": validation_results,
            "save_results": save_results,
            "supabase_url": supabase_url
        })
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))