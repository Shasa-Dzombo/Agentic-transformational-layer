import json
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import re

class MultiTableSchemaMapper:
    def __init__(self, schema_file_path: str):
        """Initialize with your multi-table database schema"""
        self.schema_file_path = schema_file_path
        self.schema = self.load_schema()
        self.tables = self.schema['database']['tables']
        self.relationships = self.schema['database']['relationships']
        self.analysis_variables = self.schema['database']['common_analysis_variables']
        
    def load_schema(self) -> Dict[str, Any]:
        """Load your database schema"""
        try:
            with open(self.schema_file_path, 'r') as f:
                schema = json.load(f)
            print(f"✅ Multi-table schema loaded: {schema['database']['name']} v{schema['database']['version']}")
            return schema
        except Exception as e:
            print(f"❌ Error loading schema: {e}")
            raise
    
    def analyze_data_for_table_assignment(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze which columns belong to which tables"""
        table_assignments = {}
        column_mapping_report = []
        
        print(f"\n🔍 Analyzing {len(df.columns)} columns for table assignment...")
        
        for table_name, table_config in self.tables.items():
            table_assignments[table_name] = []
            table_columns = table_config['columns']
            
            for df_column in df.columns:
                # Find best matching schema column
                best_match = self.find_best_column_match(df_column, table_columns)
                if best_match:
                    table_assignments[table_name].append({
                        'df_column': df_column,
                        'schema_column': best_match,
                        'table': table_name
                    })
                    column_mapping_report.append({
                        'df_column': df_column,
                        'schema_column': best_match,
                        'table': table_name,
                        'status': 'mapped'
                    })
        
        self.print_table_assignment_report(table_assignments, column_mapping_report)
        return table_assignments, column_mapping_report
    
    def find_best_column_match(self, df_column: str, table_columns: Dict[str, Any]) -> Optional[str]:
        """Find the best matching column in a table"""
        df_col_clean = self.clean_column_name(df_column)
        
        # Direct match
        for schema_col in table_columns.keys():
            if df_col_clean == self.clean_column_name(schema_col):
                return schema_col
        
        # Partial match with high confidence
        for schema_col in table_columns.keys():
            schema_col_clean = self.clean_column_name(schema_col)
            if (df_col_clean in schema_col_clean or schema_col_clean in df_col_clean) and len(df_col_clean) > 3:
                return schema_col
        
        # Semantic matching for common patterns
        semantic_matches = {
            'id': ['individual_id', 'response_id', 'health_id', 'education_id', 'employment_id', 'financial_id'],
            'birth': ['birth_date'],
            'sex': ['gender'],
            'race': ['race', 'ethnicity'],
            'income': ['income', 'salary', 'household_income'],
            'education': ['education_level', 'degree_type'],
            'height': ['height'],
            'weight': ['weight'],
            'bmi': ['bmi'],
            'blood': ['blood_pressure_systolic', 'blood_pressure_diastolic'],
            'cholesterol': ['cholesterol'],
            'glucose': ['glucose'],
        }
        
        for pattern, schema_cols in semantic_matches.items():
            if pattern in df_col_clean:
                for schema_col in schema_cols:
                    if schema_col in table_columns:
                        return schema_col
        
        return None
    
    def clean_column_name(self, column_name: str) -> str:
        """Clean column name for matching"""
        return re.sub(r'[^a-zA-Z0-9]', '', column_name.lower())
    
    def map_dataframe_to_tables(self, df: pd.DataFrame, preprocessing_results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Map preprocessed dataframe to multiple table structures"""
        table_assignments, mapping_report = self.analyze_data_for_table_assignment(df)
        mapped_tables = {}
        
        for table_name, assignments in table_assignments.items():
            if assignments:  # Only create tables that have mapped columns
                mapped_df = self.create_table_dataframe(df, table_name, assignments, preprocessing_results)
                if not mapped_df.empty:
                    mapped_tables[table_name] = mapped_df
        
        return mapped_tables
    
    def create_table_dataframe(self, df: pd.DataFrame, table_name: str, assignments: List[Dict], preprocessing_results: Dict[str, Any]) -> pd.DataFrame:
        """Create a dataframe for a specific table"""
        table_config = self.tables[table_name]
        mapped_data = {}
        
        print(f"\n📋 Creating {table_name} table...")
        
        # Handle individual_id for relational integrity
        if table_name != 'demographics' and 'individual_id' in table_config['columns']:
            # Create sequential IDs for non-demographics tables
            mapped_data['individual_id'] = range(1, len(df) + 1)
        
        # Map assigned columns
        for assignment in assignments:
            df_column = assignment['df_column']
            schema_column = assignment['schema_column']
            
            if df_column in df.columns:
                try:
                    column_config = table_config['columns'][schema_column]
                    transformed_data = self.transform_column_data(
                        df[df_column], 
                        column_config,
                        preprocessing_results
                    )
                    mapped_data[schema_column] = transformed_data
                    print(f"  ✅ {df_column} → {schema_column}")
                except Exception as e:
                    print(f"  ❌ Error mapping {df_column}: {e}")
        
        # Add required columns with defaults if missing
        for schema_column, column_config in table_config['columns'].items():
            if schema_column not in mapped_data:
                if not column_config.get('nullable', True):
                    # Add default values for required fields
                    default_value = self.get_default_value(column_config)
                    if default_value is not None:
                        mapped_data[schema_column] = [default_value] * len(df)
                        print(f"  ⚠️  Added default for required field: {schema_column}")
        
        # Create DataFrame
        if mapped_data:
            result_df = pd.DataFrame(mapped_data)
            print(f"  📊 {table_name}: {len(result_df)} rows, {len(result_df.columns)} columns")
            return result_df
        else:
            return pd.DataFrame()
    
    def transform_column_data(self, series: pd.Series, column_config: Dict[str, Any], preprocessing_results: Dict[str, Any]) -> pd.Series:
        """Transform column data according to schema requirements"""
        data_type = column_config.get('type', 'character varying')
        constraints = column_config.get('constraints', {})
        allowed_values = column_config.get('allowed_values', [])
        
        # Handle different data types
        if data_type == 'integer':
            result = pd.to_numeric(series, errors='coerce').astype('Int64')
            
            # Apply constraints
            if 'min' in constraints:
                result = result.where(result >= constraints['min'], None)
            if 'max' in constraints:
                result = result.where(result <= constraints['max'], None)
                
        elif data_type == 'numeric':
            result = pd.to_numeric(series, errors='coerce')
            
            # Apply constraints
            if 'min' in constraints:
                result = result.where(result >= constraints['min'], None)
            if 'max' in constraints:
                result = result.where(result <= constraints['max'], None)
                
        elif data_type == 'date':
            result = pd.to_datetime(series, errors='coerce').dt.date
            
        elif data_type == 'timestamp with time zone':
            result = pd.to_datetime(series, errors='coerce')
            
        elif data_type == 'character varying' or data_type == 'text':
            result = series.astype('string')
            
            # Validate allowed values
            if allowed_values:
                # Map common variations to allowed values
                result = self.map_to_allowed_values(result, allowed_values)
                
        else:
            result = series
        
        return result
    
    def map_to_allowed_values(self, series: pd.Series, allowed_values: List[str]) -> pd.Series:
        """Map series values to allowed values using fuzzy matching"""
        def find_best_match(value):
            if pd.isna(value):
                return None
            
            value_str = str(value).strip()
            
            # Direct match
            if value_str in allowed_values:
                return value_str
            
            # Case insensitive match
            for allowed in allowed_values:
                if value_str.lower() == allowed.lower():
                    return allowed
            
            # Partial match
            for allowed in allowed_values:
                if value_str.lower() in allowed.lower() or allowed.lower() in value_str.lower():
                    return allowed
            
            # Common mappings
            mappings = {
                'male': 'Male', 'm': 'Male', 'man': 'Male',
                'female': 'Female', 'f': 'Female', 'woman': 'Female',
                'white': 'White', 'caucasian': 'White',
                'black': 'Black or African American', 'african american': 'Black or African American',
                'hispanic': 'Hispanic or Latino', 'latino': 'Hispanic or Latino',
                'asian': 'Asian',
                'own': 'Own', 'rent': 'Rent', 'renting': 'Rent'
            }
            
            mapped = mappings.get(value_str.lower())
            if mapped and mapped in allowed_values:
                return mapped
            
            return value_str  # Return original if no match found
        
        return series.apply(find_best_match)
    
    def get_default_value(self, column_config: Dict[str, Any]):
        """Get appropriate default value for column type"""
        data_type = column_config.get('type')
        
        if data_type == 'integer':
            return 0
        elif data_type == 'numeric':
            return 0.0
        elif data_type in ['character varying', 'text']:
            return 'Unknown'
        elif data_type == 'timestamp with time zone':
            return pd.Timestamp.now()
        elif data_type == 'date':
            return pd.Timestamp.now().date()
        else:
            return None
    
    def print_table_assignment_report(self, table_assignments: Dict, mapping_report: List[Dict]):
        """Print detailed table assignment report"""
        print("\n" + "="*70)
        print("TABLE ASSIGNMENT REPORT")
        print("="*70)
        
        total_mapped = len([r for r in mapping_report if r['status'] == 'mapped'])
        
        print(f"📊 Total columns mapped: {total_mapped}")
        print(f"📋 Tables with data: {len([t for t, a in table_assignments.items() if a])}")
        
        for table_name, assignments in table_assignments.items():
            if assignments:
                print(f"\n🗂️  {table_name.upper()}: {len(assignments)} columns")
                for assignment in assignments:
                    print(f"   {assignment['df_column']} → {assignment['schema_column']}")