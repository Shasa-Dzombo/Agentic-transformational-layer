import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import google.generativeai as genai
import os
from datetime import datetime
import logging
import re

class AISchemaMapper:
    def __init__(self, schema_file_path: str, api_key: str = None):
        """Initialize AI-powered schema mapper"""
        self.schema_file_path = schema_file_path
        self.schema = self.load_schema()
        self.tables = self.schema['database']['tables']
        
        # Initialize Gemini AI
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key required for AI schema mapping")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_schema(self) -> Dict[str, Any]:
        """Load database schema"""
        try:
            with open(self.schema_file_path, 'r') as f:
                schema = json.load(f)
            print(f"✅ Database schema loaded: {schema['database']['name']} v{schema['database']['version']}")
            return schema
        except Exception as e:
            print(f"❌ Error loading schema: {e}")
            raise
    
    def analyze_csv_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze CSV columns to understand their content"""
        column_analysis = {}
        
        for column in df.columns:
            analysis = {
                'name': column,
                'dtype': str(df[column].dtype),
                'null_count': int(df[column].isnull().sum()),
                'null_percentage': round(df[column].isnull().sum() / len(df) * 100, 2),
                'unique_count': int(df[column].nunique()),
                'sample_values': [],
                'patterns': []
            }
            
            # Get sample non-null values
            non_null_values = df[column].dropna()
            if len(non_null_values) > 0:
                sample_size = min(5, len(non_null_values))
                analysis['sample_values'] = non_null_values.head(sample_size).tolist()
                
                # Detect patterns
                analysis['patterns'] = self.detect_column_patterns(non_null_values)
            
            column_analysis[column] = analysis
        
        return column_analysis
    
    def detect_column_patterns(self, series: pd.Series) -> List[str]:
        """Detect patterns in column data"""
        patterns = []
        sample_values = series.head(10).astype(str)
        
        # Check for common patterns
        if sample_values.str.contains(r'^\d+$').any():
            patterns.append('numeric_id')
        if sample_values.str.contains(r'anon', case=False).any():
            patterns.append('anonymized_id')
        if sample_values.str.contains(r'date|time', case=False).any():
            patterns.append('temporal')
        if sample_values.str.contains(r'male|female|sex|gender', case=False).any():
            patterns.append('gender_related')
        if sample_values.str.contains(r'age|year|birth', case=False).any():
            patterns.append('age_related')
        if sample_values.str.contains(r'income|salary|wage|money', case=False).any():
            patterns.append('financial')
        if sample_values.str.contains(r'education|school|degree', case=False).any():
            patterns.append('educational')
        if sample_values.str.contains(r'health|medical|weight|height|bmi', case=False).any():
            patterns.append('health_related')
        
        return patterns

    def create_schema_summary(self) -> str:
        """Create a concise summary of the database schema with correct table names"""
        summary_lines = []
        
        summary_lines.append("### Available Tables (use exact case):")
        for table_name, table_config in self.tables.items():
            columns = list(table_config['columns'].keys())
            key_columns = [col for col, conf in table_config['columns'].items() 
                          if conf.get('primary_key') or conf.get('foreign_key')]
            
            summary_lines.append(f"**{table_name}**: {table_config['description']}")
            summary_lines.append(f"   Key columns: {', '.join(key_columns)}")
            summary_lines.append(f"   Sample columns: {', '.join(columns[:8])}")
            summary_lines.append("")
        
        summary_lines.append("\n### IMPORTANT: Use exact table names as shown above (lowercase)")
        
        return "\n".join(summary_lines)

    def create_columns_summary(self, column_analysis: Dict[str, Any]) -> str:
        """Create a summary of CSV columns for the AI"""
        summary_lines = []
        
        # Group columns by patterns for better analysis
        pattern_groups = {}
        for col_name, analysis in column_analysis.items():
            patterns = analysis.get('patterns', ['other'])
            for pattern in patterns:
                if pattern not in pattern_groups:
                    pattern_groups[pattern] = []
                pattern_groups[pattern].append(col_name)
        
        summary_lines.append("### CSV Columns by Pattern:")
        for pattern, columns in pattern_groups.items():
            summary_lines.append(f"**{pattern}**: {', '.join(columns[:10])}")  # Limit for readability
            if len(columns) > 10:
                summary_lines.append(f"   ... and {len(columns) - 10} more")
        
        summary_lines.append("\n### Sample Column Details:")
        
        # Show detailed analysis for key columns
        key_columns = [col for col in column_analysis.keys() 
                      if any(keyword in col.lower() 
                            for keyword in ['individualid', 'gender', 'age', 'birth', 'income', 'education'])]
        
        for col in key_columns[:15]:  # Limit to prevent prompt overflow
            analysis = column_analysis[col]
            summary_lines.append(f"**{col}**:")
            summary_lines.append(f"   Type: {analysis['dtype']}, Nulls: {analysis['null_percentage']}%")
            summary_lines.append(f"   Unique values: {analysis['unique_count']}")
            summary_lines.append(f"   Sample: {analysis['sample_values']}")
            summary_lines.append(f"   Patterns: {analysis['patterns']}")
            summary_lines.append("")
        
        return "\n".join(summary_lines)
    
    def create_mapping_prompt(self, column_analysis: Dict[str, Any]) -> str:
        """Create comprehensive prompt for AI mapping with correct table names"""
        
        schema_summary = self.create_schema_summary()
        columns_summary = self.create_columns_summary(column_analysis)
        
        prompt = f"""
# AI Schema Mapping Task

You are an expert data analyst tasked with mapping CSV columns to a well-defined database schema. Your goal is to create accurate, logical mappings while avoiding conflicts and data loss.

## Database Schema Overview
{schema_summary}

## CSV Columns to Map
{columns_summary}

## Mapping Requirements

### 1. Mapping Rules
- **One-to-One Mapping**: Each CSV column should map to exactly ONE schema column in ONE table
- **No Overwrites**: Multiple CSV columns cannot map to the same schema column
- **Logical Grouping**: Related data should go to appropriate tables
- **Data Preservation**: Prioritize columns with actual data (low null percentage)
- **EXACT TABLE NAMES**: Use exactly these table names from the schema above

### 2. Priority Guidelines for APHRC HDSS Data
- **Individual Data**: res_individualid_anon, res_gender, res_datebirth, res_ethnicity → individual table
- **Household Data**: res_hhid_anon, res_hhh_* fields → household table  
- **Education Data**: edu_* fields → education table
- **Pregnancy Data**: gbs_*, pge_* fields → pregnancy table
- **Income Data**: iga_* fields → livelihoods table
- **Vaccination Data**: vac_* fields → vaccination table
- **Migration Data**: los_* fields are statistical, avoid mapping to individual records

### 3. Conflict Resolution
If multiple CSV columns could map to the same schema column:
- Choose the column with the BEST data quality (fewest nulls)
- Choose the most SPECIFIC/RELEVANT column name
- Prefer PRIMARY identifiers over secondary ones

### 4. Special Handling for Your Data
- **res_individualid_anon**: Primary individual identifier → individual.individual_id
- **res_hhid_anon**: Household identifier → household.household_id
- **res_gender**: Individual gender → individual.sex
- **res_datebirth**: Birth date → individual.date_of_birth
- **los_* columns**: These are statistical loss-to-follow-up data, NOT individual records - DO NOT MAP
- **vac_* columns**: Vaccination status → vaccination table
- **edu_* columns**: Education data → education table

## Output Format
Provide your mapping as a JSON object with this exact structure and EXACT table names:

```json
{{
  "table_mappings": {{
    "individual": [
      {{
        "csv_column": "res_individualid_anon",
        "schema_column": "individual_id", 
        "confidence": 0.95,
        "reasoning": "Primary individual identifier with unique values"
      }},
      {{
        "csv_column": "res_gender",
        "schema_column": "sex",
        "confidence": 0.95,
        "reasoning": "Individual gender data maps to sex field"
      }}
    ],
    "household": [
      {{
        "csv_column": "res_hhid_anon",
        "schema_column": "household_id",
        "confidence": 0.95,
        "reasoning": "Household identifier"
      }}
    ],
    "education": [
      {{
        "csv_column": "edu_everschool",
        "schema_column": "has_ever_attended_school",
        "confidence": 0.90,
        "reasoning": "Ever attended school status"
      }}
    ]
  }},
  "unmapped_columns": [
    {{
      "csv_column": "los_chddth_allsex_anyage",
      "reasoning": "Statistical aggregation data, not suitable for individual records"
    }}
  ],
  "mapping_summary": {{
    "total_csv_columns": 239,
    "mapped_columns": 15,
    "unmapped_columns": 224,
    "confidence_score": 0.88
  }}
}}
```

## Critical Requirements
- **Table Names**: Use ONLY these exact names: {list(self.tables.keys())}
- **Quality over Quantity**: Map only high-confidence matches
- **Avoid Statistical Data**: Do not map los_* aggregated columns to individual records
- **Primary Keys**: Always map primary identifiers (res_individualid_anon, res_hhid_anon)

Please analyze the data carefully and provide intelligent, conflict-free mappings using the exact table names specified.
"""
        return prompt

    def get_ai_mapping(self, column_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-generated mapping"""
        prompt = self.create_mapping_prompt(column_analysis)
        
        try:
            print("🤖 Generating AI schema mapping...")
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            response_text = response.text
            
            # Find JSON in response (between ```json and ```
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No valid JSON found in AI response")
            
            # Parse JSON
            mapping_result = json.loads(json_str)
            
            print(f"✅ AI mapping completed!")
            print(f"   Mapped columns: {mapping_result['mapping_summary']['mapped_columns']}")
            print(f"   Confidence score: {mapping_result['mapping_summary']['confidence_score']}")
            
            return mapping_result
            
        except Exception as e:
            print(f"❌ Error getting AI mapping: {e}")
            if 'response_text' in locals():
                print(f"Response text: {response_text[:500]}...")
            raise
    
    def validate_ai_mapping(self, mapping_result: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the AI-generated mapping with case-insensitive table names"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Create lowercase mapping of table names for case-insensitive comparison
        schema_tables_lower = {name.lower(): name for name in self.tables.keys()}
        
        # Normalize table names in mapping result to lowercase
        normalized_mappings = {}
        for table_name, mappings in mapping_result['table_mappings'].items():
            normalized_table_name = table_name.lower()
            if normalized_table_name in schema_tables_lower:
                # Use the correct case from schema
                correct_table_name = schema_tables_lower[normalized_table_name]
                normalized_mappings[correct_table_name] = mappings
            else:
                validation_results['errors'].append(f"Invalid table name: {table_name}")
                validation_results['valid'] = False
        
        # Update the mapping result with normalized table names
        mapping_result['table_mappings'] = normalized_mappings
        
        # Check for conflicts (same schema column mapped multiple times)
        schema_column_usage = {}
        for table_name, mappings in normalized_mappings.items():
            for mapping in mappings:
                schema_col = f"{table_name}.{mapping['schema_column']}"
                if schema_col in schema_column_usage:
                    validation_results['errors'].append(
                        f"Conflict: {schema_col} mapped to both {schema_column_usage[schema_col]} and {mapping['csv_column']}"
                    )
                    validation_results['valid'] = False
                else:
                    schema_column_usage[schema_col] = mapping['csv_column']
        
        # Check CSV column existence
        for table_name, mappings in normalized_mappings.items():
            for mapping in mappings:
                if mapping['csv_column'] not in df.columns:
                    validation_results['errors'].append(
                        f"CSV column '{mapping['csv_column']}' not found in dataframe"
                    )
                    validation_results['valid'] = False
        
        # Check schema column validity
        for table_name, mappings in normalized_mappings.items():
            if table_name not in self.tables:
                validation_results['errors'].append(f"Invalid table name: {table_name}")
                validation_results['valid'] = False
                continue
                
            for mapping in mappings:
                if mapping['schema_column'] not in self.tables[table_name]['columns']:
                    validation_results['errors'].append(
                        f"Invalid schema column: {table_name}.{mapping['schema_column']}"
                    )
                    validation_results['valid'] = False
        
        # Statistics
        validation_results['statistics'] = {
            'total_tables_used': len(normalized_mappings),
            'total_mappings': sum(len(mappings) for mappings in normalized_mappings.values()),
            'conflict_count': len([err for err in validation_results['errors'] if 'Conflict' in err])
        }
        
        return validation_results
    
    def map_dataframe_to_tables(self, df: pd.DataFrame, preprocessing_results: Dict[str, Any] = None) -> Dict[str, pd.DataFrame]:
        """Main method to map DataFrame to tables using AI"""
        print(f"\n🤖 Starting AI-powered schema mapping for {len(df.columns)} columns...")
        
        # Step 1: Analyze CSV columns
        column_analysis = self.analyze_csv_columns(df)
        print(f"📊 Column analysis completed")
        
        # Step 2: Get AI mapping
        mapping_result = self.get_ai_mapping(column_analysis)
        
        # Step 3: Validate mapping
        validation = self.validate_ai_mapping(mapping_result, df)
        
        if not validation['valid']:
            print(f"❌ AI mapping validation failed:")
            for error in validation['errors']:
                print(f"   • {error}")
            raise ValueError("AI mapping validation failed")
        
        # Step 4: Create tables
        mapped_tables = self.create_tables_from_ai_mapping(df, mapping_result, preprocessing_results)
        
        # Step 5: Print summary
        self.print_mapping_summary(mapping_result, mapped_tables)
        
        return mapped_tables
    
    def create_tables_from_ai_mapping(self, df: pd.DataFrame, mapping_result: Dict[str, Any], preprocessing_results: Dict[str, Any] = None) -> Dict[str, pd.DataFrame]:
        """Create database tables from AI mapping"""
        mapped_tables = {}
        
        for table_name, mappings in mapping_result['table_mappings'].items():
            if not mappings:
                continue
                
            print(f"\n📋 Creating {table_name} table from AI mapping...")
            
            table_data = {}
            table_config = self.tables[table_name]
            
            # Map assigned columns
            for mapping in mappings:
                csv_column = mapping['csv_column']
                schema_column = mapping['schema_column']
                confidence = mapping.get('confidence', 1.0)
                
                if csv_column in df.columns:
                    # Transform data according to schema
                    column_config = table_config['columns'][schema_column]
                    transformed_data = self.transform_column_data(
                        df[csv_column], 
                        column_config,
                        preprocessing_results or {}
                    )
                    table_data[schema_column] = transformed_data
                    print(f"   ✅ {csv_column} → {schema_column} (confidence: {confidence:.2f})")
            
            # Add required default columns
            for schema_column, column_config in table_config['columns'].items():
                if schema_column not in table_data:
                    if not column_config.get('nullable', True):
                        # Add default for required columns
                        default_value = self.get_default_value(column_config)
                        if default_value is not None:
                            table_data[schema_column] = [default_value] * len(df)
                            print(f"   ⚠️  Added default for required field: {schema_column}")
                    elif column_config.get('default') == 'now()':
                        # Add timestamp defaults
                        table_data[schema_column] = [pd.Timestamp.now()] * len(df)
                        print(f"   🕐 Added timestamp default: {schema_column}")
            
            # Create DataFrame
            if table_data:
                table_df = pd.DataFrame(table_data)
                mapped_tables[table_name] = table_df
                print(f"   📊 {table_name}: {len(table_df)} rows, {len(table_df.columns)} columns")
        
        return mapped_tables
    
    def transform_column_data(self, series: pd.Series, column_config: Dict[str, Any], preprocessing_results: Dict[str, Any]) -> pd.Series:
        """Transform column data according to schema (reuse from existing mapper)"""
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
            
            # Common mappings for your data
            mappings = {
                'male': 'Male', 'm': 'Male', '1': 'Male',
                'female': 'Female', 'f': 'Female', '2': 'Female',
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
            return 1  # For IDs, start at 1
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
    
    def print_mapping_summary(self, mapping_result: Dict[str, Any], mapped_tables: Dict[str, pd.DataFrame]):
        """Print comprehensive mapping summary"""
        print("\n" + "="*80)
        print("🤖 AI SCHEMA MAPPING SUMMARY")
        print("="*80)
        
        summary = mapping_result['mapping_summary']
        print(f"📊 Total CSV columns analyzed: {summary['total_csv_columns']}")
        print(f"✅ Successfully mapped columns: {summary['mapped_columns']}")
        print(f"❌ Unmapped columns: {summary['unmapped_columns']}")
        print(f"🎯 Overall confidence score: {summary['confidence_score']}")
        
        print(f"\n📋 Tables created: {len(mapped_tables)}")
        for table_name, table_df in mapped_tables.items():
            print(f"   {table_name}: {len(table_df)} rows, {len(table_df.columns)} columns")
        
        print(f"\n🗂️  Detailed Mappings:")
        for table_name, mappings in mapping_result['table_mappings'].items():
            if mappings:
                print(f"\n{table_name.upper()}:")
                for mapping in mappings:
                    confidence = mapping.get('confidence', 1.0)
                    print(f"   ✅ {mapping['csv_column']} → {mapping['schema_column']} (confidence: {confidence:.2f})")
        
        if mapping_result.get('unmapped_columns'):
            print(f"\n❌ Unmapped columns (sample):")
            for unmapped in mapping_result['unmapped_columns'][:10]:  # Show first 10
                print(f"   • {unmapped['csv_column']}: {unmapped['reasoning']}")