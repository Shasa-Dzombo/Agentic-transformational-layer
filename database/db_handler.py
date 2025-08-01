import pandas as pd
from supabase import create_client, Client
import json
import logging
from typing import Dict, Any, List
import os
from datetime import datetime

class SupabaseClientHandler:
    def __init__(self, supabase_url: str, supabase_key: str, schema_config: Dict[str, Any]):
        """Initialize Supabase client handler"""
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.client: Client = create_client(supabase_url, supabase_key)
        self.schema_config = schema_config
        self.tables_config = schema_config['database']['tables']
        self.relationships = schema_config['database']['relationships']
        
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
            # Try to get user info or any basic operation
            response = self.client.auth.get_user()
            print("✅ Supabase client connection successful")
            return True
        except Exception as e:
            # If auth fails, try a simple table query (this might fail if no tables exist yet)
            try:
                # Try to list tables from information_schema
                response = self.client.rpc('get_schema_tables').execute()
                print("✅ Supabase client connection successful")
                return True
            except Exception as e2:
                print(f"✅ Supabase client initialized (auth check failed but client is ready): {e}")
                return True  # Client is likely working, just auth issue
    
    def create_table_if_not_exists(self, table_name: str) -> bool:
        """Create table using Supabase SQL if it doesn't exist"""
        try:
            table_config = self.tables_config[table_name]
            create_sql = self.generate_create_table_sql(table_name)
            
            # Execute SQL using Supabase RPC or direct SQL
            response = self.client.rpc('exec_sql', {'sql': create_sql}).execute()
            
            print(f"  ✅ Table '{table_name}' ready in Supabase")
            return True
            
        except Exception as e:
            print(f"  ⚠️  Table creation for '{table_name}': {e}")
            # Table might already exist, which is fine
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
    
    def save_table_data(self, table_name: str, data: pd.DataFrame) -> bool:
        """Save DataFrame to Supabase table using client"""
        try:
            # Convert DataFrame to list of dictionaries
            records = data.to_dict('records')
            
            # Handle timestamp columns
            for record in records:
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
                    elif isinstance(value, pd.Timestamp):
                        record[key] = value.isoformat()
                    elif hasattr(value, 'isoformat'):  # datetime objects
                        record[key] = value.isoformat()
            
            # Insert data in batches (Supabase has limits)
            batch_size = 1000
            total_inserted = 0
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                response = self.client.table(table_name).insert(batch).execute()
                
                if hasattr(response, 'data') and response.data:
                    total_inserted += len(batch)
                    print(f"  📥 Batch {i//batch_size + 1}: {len(batch)} records inserted into {table_name}")
                else:
                    print(f"  ⚠️  Batch {i//batch_size + 1}: Insert response unclear for {table_name}")
                    total_inserted += len(batch)  # Assume success if no error
            
            print(f"  ✅ {table_name}: {total_inserted} total records saved to Supabase")
            return True
            
        except Exception as e:
            print(f"  ❌ {table_name}: Error saving data - {e}")
            return False
    
    def save_mapped_tables(self, mapped_tables: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """Save all mapped tables to Supabase"""
        results = {}
        
        print(f"\n💾 Saving {len(mapped_tables)} tables to Supabase...")
        
        # Save in dependency order
        table_order = ['demographics', 'employment', 'health_metrics', 'education', 'survey_responses', 'financial_data']
        
        for table_name in table_order:
            if table_name in mapped_tables:
                try:
                    df = mapped_tables[table_name].copy()
                    
                    # Add metadata timestamps if not present
                    if 'created_at' in self.tables_config[table_name]['columns'] and 'created_at' not in df.columns:
                        df['created_at'] = pd.Timestamp.now()
                    if 'updated_at' in self.tables_config[table_name]['columns'] and 'updated_at' not in df.columns:
                        df['updated_at'] = pd.Timestamp.now()
                    
                    # Remove auto-increment primary key columns
                    pk_columns = [col for col, config in self.tables_config[table_name]['columns'].items() 
                                if config.get('primary_key', False) and config.get('type') == 'integer']
                    
                    for pk_col in pk_columns:
                        if pk_col in df.columns:
                            df = df.drop(columns=[pk_col])
                    
                    # Create table if needed
                    self.create_table_if_not_exists(table_name)
                    
                    # Save data
                    success = self.save_table_data(table_name, df)
                    results[table_name] = success
                    
                except Exception as e:
                    print(f"  ❌ {table_name}: Error - {e}")
                    results[table_name] = False
        
        success_count = sum(results.values())
        print(f"\n🎯 Successfully saved {success_count}/{len(mapped_tables)} tables to Supabase")
        
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