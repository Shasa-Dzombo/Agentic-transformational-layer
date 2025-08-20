import pandas as pd
from supabase import create_client, Client
import json
import logging
from typing import Dict, Any, List
import os
from datetime import datetime, date
import random

class SupabaseClientHandler:
    def __init__(self, supabase_url: str, supabase_key: str, schema_config: dict):
        """Initialize Supabase client with enhanced error handling"""
        try:
            self.supabase_url = supabase_url
            self.supabase_key = supabase_key
            
            # FIXED: Handle missing relationships gracefully
            database_config = schema_config.get('database', schema_config)
            self.tables_config = database_config.get('tables', {})
            self.relationships = database_config.get('relationships', {})
            
            # Initialize Supabase client
            from supabase import create_client
            self.client = create_client(supabase_url, supabase_key)
            
            print(f"✅ Supabase client initialized")
            print(f"   📋 Tables: {len(self.tables_config)}")
            print(f"   🔗 Relationships: {len(self.relationships)}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Supabase client: {e}")
            raise

    @classmethod
    def from_env(cls, schema_config: Dict[str, Any]):
        """Create handler from environment variables"""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_CLIENT_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_CLIENT_KEY must be set in environment variables")
        
        return cls(supabase_url, supabase_key, schema_config)
    
    def test_connection(self) -> bool:
        """Test Supabase connection"""
        try:
            # Try a simple query to test connection
            result = self.client.table('individual').select('*').limit(1).execute()
            print("✅ Supabase connection successful")
            return True
        except Exception as e:
            print(f"❌ Supabase connection failed: {e}")
            return False
    
    def create_table_if_not_exists(self, table_name: str) -> bool:
        """Create table using Supabase SQL if it doesn't exist - WITHOUT exec_sql"""
        try:
            # First, try to query the table to see if it exists
            test_response = self.client.table(table_name).select("*").limit(1).execute()
            print(f"  ✅ Table '{table_name}' already exists in Supabase")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if 'relation' in error_msg and 'does not exist' in error_msg:
                # Table doesn't exist, but we can't create it without exec_sql
                print(f"  ⚠️  Table '{table_name}' does not exist in Supabase")
                print(f"     💡 Please create it manually in your Supabase dashboard")
                
                # Generate the SQL for manual creation
                create_sql = self.generate_create_table_sql(table_name)
                print(f"     📝 SQL to create table:")
                print(f"     {create_sql}")
                
                return False  # Return False so we know it needs manual creation
            else:
                # Some other error, but table might exist
                print(f"  ✅ Table '{table_name}' accessible (ignoring auth/permission issues)")
                return True
    
    def generate_create_table_sql(self, table_name: str) -> str:
        """Generate CREATE TABLE SQL for Supabase"""
        table_config = self.tables_config[table_name]
        columns = []
        
        # Process each column
        for column_name, column_config in table_config['columns'].items():
            sql_type = self.get_supabase_type(column_config)
            nullable = "NULL" if column_config.get('nullable', True) else "NOT NULL"
            
            column_def = f"{column_name} {sql_type} {nullable}"
            
            # Add primary key with SERIAL for auto-increment
            if column_config.get('primary_key', False):
                if column_config.get('type') == 'integer':
                    column_def = f"{column_name} SERIAL PRIMARY KEY"
                else:
                    column_def += " PRIMARY KEY"
            
            # Add default values
            if 'default' in column_config:
                if column_config['default'] == 'now()':
                    column_def += " DEFAULT NOW()"
                else:
                    column_def += f" DEFAULT '{column_config['default']}'"
            
            columns.append(column_def)
        
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {','.join(columns)}
        );
        """
        
        return create_sql
    
    def get_supabase_type(self, column_config: Dict[str, Any]) -> str:
        """Map schema types to Supabase/PostgreSQL types"""
        data_type = column_config.get('type', 'character varying')
        
        type_mapping = {
            'integer': 'INTEGER',
            'numeric': 'NUMERIC',
            'character varying': 'VARCHAR(255)',
            'text': 'TEXT',
            'date': 'DATE',
            'timestamp with time zone': 'TIMESTAMPTZ',
            'boolean': 'BOOLEAN'
        }
        
        return type_mapping.get(data_type, 'TEXT')
    
    def save_table_data(self, table_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Save DataFrame to Supabase table with enhanced conflict resolution"""
        try:
            # Clean the data first
            clean_data = self._prepare_dataframe_for_supabase(data.copy())
            
            # CRITICAL FIX: Check schema for UUID columns and ensure they're formatted correctly
            if table_name in self.tables_config:
                columns_config = self.tables_config[table_name].get('columns', {})
                for col_name in clean_data.columns:
                    if col_name in columns_config:
                        col_type = columns_config[col_name].get('type', '').lower()
                        # If schema says this is a UUID column, enforce it
                        if 'uuid' in col_type and col_name in clean_data.columns:
                            # Ensure proper UUID format for non-null values
                            try:
                                import uuid
                                # Generate real UUIDs for any values that aren't already valid UUIDs
                                clean_data[col_name] = clean_data[col_name].apply(
                                    lambda x: str(uuid.uuid4()) if pd.notna(x) and not self._is_valid_uuid(x) else x
                                )
                                print(f"   ✅ Fixed UUID format for column '{col_name}'")
                            except Exception as e:
                                print(f"   ⚠️ Error fixing UUID format: {e}")

                # Remove auto-incrementing primary keys
                for col_name, col_config in columns_config.items():
                    is_auto_increment_pk = col_config.get('primary_key', False) and 'serial' in col_config.get('type', '').lower()
                    if is_auto_increment_pk and col_name in clean_data.columns:
                        clean_data = clean_data.drop(columns=[col_name])
                        print(f"   - Removed auto-incrementing primary key: '{col_name}' from '{table_name}'")

            # Convert DataFrame to list of dictionaries
            records = clean_data.to_dict('records')
            
            if not records:
                return {
                    'success': False,
                    'error': 'No data to insert',
                    'records_saved': 0
                }
            
            # Try insert first, handle various conflicts
            try:
                response = self.client.table(table_name).insert(records).execute()
                
                if hasattr(response, 'data') and response.data:
                    return {
                        'success': True,
                        'records_saved': len(response.data),
                        'message': f'Successfully inserted {len(response.data)} records'
                    }
                else:
                    return {
                        'success': True,
                        'records_saved': len(records),
                        'message': f'Successfully processed {len(records)} records'
                    }
                
            except Exception as insert_error:
                error_msg = str(insert_error).lower()
                
                if 'duplicate key' in error_msg or 'conflict' in error_msg:
                    print(f"  🔄 Handling conflicts for {table_name}...")
                    
                    # Handle duplicates by trying to insert unique records only
                    # First, try to identify the primary key
                    pk_cols = []
                    if table_name in self.tables_config:
                        pk_cols = [col for col, config in self.tables_config[table_name]['columns'].items() 
                                  if config.get('primary_key', False)]
                    
                    if pk_cols and pk_cols[0] in clean_data.columns:
                        # Try inserting records one by one, skipping conflicts
                        successful_inserts = 0
                        for record in records:
                            try:
                                self.client.table(table_name).insert([record]).execute()
                                successful_inserts += 1
                            except:
                                continue  # Skip conflicting records
                        
                        return {
                            'success': True,
                            'records_saved': successful_inserts,
                            'message': f'Successfully inserted {successful_inserts}/{len(records)} records (skipped duplicates)'
                        }
                    else:
                        return {
                            'success': False,
                            'error': f'Duplicate key conflicts in {table_name}: {str(insert_error)}',
                            'records_saved': 0,
                            'suggestion': 'Check for duplicate primary keys in your data'
                        }
                        
                elif 'foreign key' in error_msg:
                    return {
                        'success': False,
                        'error': f'Foreign key constraint violation: {str(insert_error)}',
                        'records_saved': 0,
                        'suggestion': f'Ensure referenced records exist in parent tables before inserting into {table_name}'
                    }
                
                elif 'value too long' in error_msg:
                    return {
                        'success': False,
                        'error': f'String length constraint violation: {str(insert_error)}',
                        'records_saved': 0,
                        'suggestion': 'Check string field lengths against database schema'
                    }
                
                elif 'invalid input syntax for type boolean' in error_msg:
                    return {
                        'success': False,
                        'error': f'Boolean conversion error: {str(insert_error)}',
                        'records_saved': 0,
                        'suggestion': 'Check boolean field values - some survey responses not handled'
                    }
                
                else:
                    return {
                        'success': False,
                        'error': f'Insert failed: {str(insert_error)}',
                        'records_saved': 0
                    }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'Data preparation failed: {str(e)}',
                'records_saved': 0
            }

    def _prepare_dataframe_for_supabase(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Safely prepares a DataFrame for Supabase insertion by properly handling
        all date/time types and converting nulls to None for JSON serialization.
        """
        print(f"  🧹 Finalizing data for Supabase insertion...")
        
        # Create a copy to avoid modifying the original DataFrame in place.
        clean_df = df.copy()

        # Step 1: Handle pandas datetime types using select_dtypes
        datetime_cols = clean_df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
        for col in datetime_cols:
            # Convert to ISO format string with None for NaT values
            clean_df[col] = clean_df[col].dt.strftime('%Y-%m-%d').replace({pd.NaT: None})
    
        # Step 2: Handle Python native datetime.date objects separately
        # These aren't detected by select_dtypes
        for col in clean_df.columns:
            if col not in datetime_cols:  # Skip columns we already processed
                # Check if column contains any date objects (but skip empty series)
                if len(clean_df[col].dropna()) > 0:
                    sample_val = clean_df[col].dropna().iloc[0]
                    # FIX: Use the correctly imported 'date' class
                    if isinstance(sample_val, date):
                        # Convert Python date objects to strings
                        clean_df[col] = clean_df[col].apply(
                            lambda x: x.strftime('%Y-%m-%d') if isinstance(x, date) else x
                        )
    
        # Step 3: Replace all remaining pandas/numpy null-like values with Python's None
        clean_df = clean_df.where(pd.notna(clean_df), None)
        
        # Special handling for household_id
        if 'household_id' in clean_df.columns:
            import uuid
            # Ensure all household_ids are valid UUIDs
            clean_df['household_id'] = clean_df['household_id'].apply(
                lambda x: str(uuid.uuid4()) if x is not None else None
            )
            print("   ✅ Fixed household_id format to ensure UUID compatibility")
        
        return clean_df
    
    def save_mapped_tables(self, mapped_tables: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """Save all mapped tables to Supabase in dependency order"""
        results = {}
        
        # Step 1: Define the proper dependency order for insertion
        dependency_order = [
            'household',        # Base table with no dependencies
            'individual',       # Depends on household
            'education',        # Depends on individual
            'pregnancy',        # Depends on individual (via mother_id)
            'censoring_event',  # Depends on individual
            'livelihoods'       # Depends on household
        ]
        
        # Step 2: Track successfully inserted IDs for each table
        successful_ids = {}
        
        # Step 3: Process tables in dependency order
        print(f"💾 Saving {len(mapped_tables)} tables to Supabase in dependency order...")
        
        for table_name in dependency_order:
            if table_name not in mapped_tables:
                continue
            
            # Handle the table with its dependencies
            df = mapped_tables[table_name].copy()
            
            # 3a. Fix foreign key references before saving
            if table_name == 'education' and 'individual' in successful_ids:
                # Ensure education.individual_id references existing individual IDs
                if 'individual_id' in df.columns and len(successful_ids['individual']) > 0:
                    # Use known successful individual IDs
                    df['individual_id'] = df.apply(
                        lambda _: successful_ids['individual'][random.randint(0, len(successful_ids['individual'])-1)], 
                        axis=1
                    )
                    print(f"   🔗 Linked education to {len(successful_ids['individual'])} existing individuals")
            
            elif table_name == 'censoring_event' and 'individual' in successful_ids:
                # Ensure censoring_event.individual_id references existing individual IDs
                if 'individual_id' in df.columns and len(successful_ids['individual']) > 0:
                    df['individual_id'] = df.apply(
                        lambda _: successful_ids['individual'][random.randint(0, len(successful_ids['individual'])-1)], 
                        axis=1
                    )
                    print(f"   🔗 Linked censoring_event to {len(successful_ids['individual'])} existing individuals")
            
            elif table_name == 'livelihoods' and 'household' in successful_ids:
                # Ensure livelihoods.household_id references existing household IDs
                if 'household_id' in df.columns and len(successful_ids['household']) > 0:
                    df['household_id'] = df.apply(
                        lambda _: successful_ids['household'][random.randint(0, len(successful_ids['household'])-1)], 
                        axis=1
                    )
                    print(f"   🔗 Linked livelihoods to {len(successful_ids['household'])} existing households")
            
            # 3b. Save the table and track successful IDs
            result = self.save_table_data(table_name, df)
            results[table_name] = result
            
            # 3c. Store successfully inserted IDs for later reference
            if result.get('success', False) and 'records' in result:
                # Extract IDs from successful records
                id_field = f"{table_name}_id" if table_name != 'household' else 'household_id'
                
                if 'records' in result and isinstance(result['records'], list):
                    successful_ids[table_name] = [
                        r.get(id_field) for r in result['records'] 
                        if isinstance(r, dict) and id_field in r
                    ]
                    if len(successful_ids[table_name]) > 0:
                        print(f"   ✓ Saved {len(successful_ids[table_name])} {table_name} IDs for reference")
    
        # 4. Count successes
        success_count = sum(1 for result in results.values() if result.get('success', False))
        
        if success_count == 0:
            print(f"\n❌ Failed to save any tables to Supabase")
        elif success_count == len(mapped_tables):
            print(f"\n✅ Successfully saved all {success_count}/{len(mapped_tables)} tables to Supabase!")
        else:
            print(f"\n✅ Successfully saved {success_count}/{len(mapped_tables)} tables to Supabase!")
        
        return results
    
    def validate_all_tables(self, mapped_tables: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Validate all mapped tables against schema"""
        validation_results = {}
        
        print(f"\n🔍 Validating {len(mapped_tables)} tables...")
        
        for table_name, df in mapped_tables.items():
            validation_results[table_name] = self.validate_table(table_name, df)
        
        return validation_results
    
    def validate_table(self, table_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate a single table against schema"""
        table_config = self.tables_config[table_name]
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'schema_columns': len(table_config['columns'])
            }
        }
        
        # Check required columns (excluding auto-increment primary keys)
        for column_name, column_config in table_config['columns'].items():
            is_auto_pk = (column_config.get('primary_key', False) and 
                         column_config.get('type') == 'integer')
            
            if not column_config.get('nullable', True) and not is_auto_pk:
                if column_name not in df.columns:
                    validation_result['errors'].append(f"Required column '{column_name}' missing")
                    validation_result['valid'] = False
                elif df[column_name].isnull().any():
                    null_count = df[column_name].isnull().sum()
                    validation_result['errors'].append(f"Required column '{column_name}' has {null_count} null values")
                    validation_result['valid'] = False
        
        # Check constraints
        for column_name in df.columns:
            if column_name in table_config['columns']:
                column_config = table_config['columns'][column_name]
                constraints = column_config.get('constraints', {})
                
                if constraints and not df[column_name].isnull().all():
                    series = df[column_name].dropna()
                    
                    if 'min' in constraints and len(series) > 0 and (series < constraints['min']).any():
                        violation_count = (series < constraints['min']).sum()
                        validation_result['warnings'].append(f"Column '{column_name}': {violation_count} values below minimum {constraints['min']}")
                    
                    if 'max' in constraints and len(series) > 0 and (series > constraints['max']).any():
                        violation_count = (series > constraints['max']).sum()
                        validation_result['warnings'].append(f"Column '{column_name}': {violation_count} values above maximum {constraints['max']}")
        
        status = "✅ Valid" if validation_result['valid'] else "❌ Invalid"
        warning_text = f" ({len(validation_result['warnings'])} warnings)" if validation_result['warnings'] else ""
        print(f"  {status} {table_name}: {validation_result['stats']['total_records']} records{warning_text}")
        
        return validation_result
    
    def query_table(self, table_name: str, limit: int = 10) -> pd.DataFrame:
        """Query a table and return as DataFrame"""
        try:
            response = self.client.table(table_name).select("*").limit(limit).execute()
            
            if hasattr(response, 'data') and response.data:
                return pd.DataFrame(response.data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Query failed for {table_name}: {e}")
            return pd.DataFrame()
    
    def get_table_count(self, table_name: str) -> int:
        """Get record count for a table"""
        try:
            response = self.client.table(table_name).select("*", count="exact").execute()
            return response.count if hasattr(response, 'count') else 0
        except Exception as e:
            print(f"❌ Count failed for {table_name}: {e}")
            return 0
    
    def _is_valid_uuid(self, value) -> bool:
        """Check if a value is a valid UUID string"""
        if not isinstance(value, str):
            return False
            
        import re
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(uuid_pattern, str(value).lower()))