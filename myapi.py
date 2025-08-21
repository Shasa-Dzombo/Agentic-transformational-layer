from fastapi import FastAPI

app = FastAPI()

# @app.get("/")
# async def read_root():
#     return {"message": "Hello, World!"}


# --- Modular pipeline state ---
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
import uuid

# In-memory session store (for demo; use Redis or DB for production)
session_store = {}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV and start a session."""
    session_id = str(uuid.uuid4())
    temp_path = f"temp_{session_id}_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    df = load_csv_data(temp_path)
    if df is None:
        os.remove(temp_path)
        raise HTTPException(status_code=400, detail="Failed to load CSV data.")
    session_store[session_id] = {"raw_df": df, "temp_path": temp_path}
    return {"session_id": session_id, "shape": df.shape}

@app.post("/smart-filter")
async def smart_filter(session_id: str = Form(...), use_ai_filtering: bool = Form(True)):
    """Apply smart filter to uploaded data."""
    session = session_store.get(session_id)
    if not session or "raw_df" not in session:
        raise HTTPException(status_code=404, detail="Session not found or CSV not uploaded.")
    filtered_df = apply_smart_filter(session["raw_df"], use_ai=use_ai_filtering)
    session["filtered_df"] = filtered_df
    return {"filtered_shape": filtered_df.shape}

@app.post("/preprocess")
async def preprocess(session_id: str = Form(...)):
    """Run preprocessing workflow on filtered data."""
    session = session_store.get(session_id)
    if not session or "filtered_df" not in session:
        raise HTTPException(status_code=404, detail="Session not found or smart filter not applied.")
    initial_state = {"df": session["filtered_df"]}
    app_graph = builder.compile()
    final_state = app_graph.invoke(initial_state)
    processed_df = final_state["df"]
    session["processed_df"] = processed_df
    return {"processed_shape": processed_df.shape}

@app.post("/schema-map")
async def schema_map(session_id: str = Form(...)):
    """Map processed data to tables."""
    session = session_store.get(session_id)
    if not session or "processed_df" not in session:
        raise HTTPException(status_code=404, detail="Session not found or preprocessing not done.")
    SCHEMA_FILE_PATH = os.path.join("database", "database_schema.json")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
    mapper = EnhancedAISchemaMapper(SCHEMA_FILE_PATH, api_key)
    preprocessing_results = {
        "null_suggestions": "Preprocessing completed",
        "duplicate_suggestions": "No duplicates found",
        "type_suggestions": "Types optimized",
        "type_analysis": {}
    }
    mapped_tables = mapper.map_dataframe_to_tables(session["processed_df"], preprocessing_results)
    session["mapped_tables"] = mapped_tables
    return {"tables": {k: {"rows": len(v), "columns": len(v.columns)} for k, v in mapped_tables.items()}}

@app.post("/save-db")
async def save_db(session_id: str = Form(...)):
    """Validate and save mapped tables to Supabase."""
    session = session_store.get(session_id)
    if not session or "mapped_tables" not in session:
        raise HTTPException(status_code=404, detail="Session not found or schema mapping not done.")
    SCHEMA_FILE_PATH = os.path.join("database", "database_schema.json")
    supabase_url, supabase_key = setup_supabase_client()
    with open(SCHEMA_FILE_PATH, 'r') as f:
        schema_config = json.load(f)
    db_handler = SupabaseClientHandler(supabase_url, supabase_key, schema_config)
    if not db_handler.test_connection():
        raise HTTPException(status_code=500, detail="Supabase connection failed.")
    validation_results = db_handler.validate_all_tables(session["mapped_tables"])
    save_results = db_handler.save_mapped_tables(session["mapped_tables"])
    return {
        "validation": validation_results,
        "save_results": save_results,
        "supabase_url": supabase_url
    }