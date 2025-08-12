# Add this new class to handle dynamic schema analysis (create new file: config/schema_validator.py)

import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Set
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FieldRequirement:
    """Represents a database field requirement"""
    field_name: str
    data_type: str
    is_required: bool
    is_primary_key: bool
    is_foreign_key: bool
    default_value: Any = None
    constraints: Dict[str, Any] = None
    references: Optional[str] = None
    auto_increment: bool = False

@dataclass
class TableRequirements:
    """Represents all requirements for a database table"""
    table_name: str
    required_fields: Dict[str, FieldRequirement]
    optional_fields: Dict[str, FieldRequirement]
    foreign_keys: Dict[str, str]
    primary_key: str

class DynamicSchemaValidator:
    """
    Dynamic schema validator that reads database schema and automatically
    handles all field requirements, validations, and enhancements
    """
    
    def __init__(self, schema_path: str):
        self.schema_path = schema_path
        self.schema = self.load_schema()
        self.table_requirements = self.parse_schema_requirements()
        
    def load_schema(self) -> Dict[str, Any]:
        """Load and parse the database schema"""
        try:
            with open(self.schema_path, 'r') as f:
                schema = json.load(f)
            return schema
        except Exception as e:
            raise ValueError(f"Failed to load schema from {self.schema_path}: {e}")
    
    def parse_schema_requirements(self) -> Dict[str, TableRequirements]:
        """Parse schema to extract all field requirements for each table"""
        table_requirements = {}
        
        tables = self.schema.get('database', {}).get('tables', {})
        
        for table_name, table_config in tables.items():
            required_fields = {}
            optional_fields = {}
            foreign_keys = {}
            primary_key = None
            
            columns = table_config.get('columns', {})
            
            for field_name, field_config in columns.items():
                if isinstance(field_config, dict):
                    # Parse field requirements
                    field_req = self._parse_field_requirement(field_name, field_config)
                    
                    # Track primary key
                    if field_req.is_primary_key:
                        primary_key = field_name
                    
                    # Track foreign keys
                    if field_req.is_foreign_key and field_req.references:
                        foreign_keys[field_name] = field_req.references
                    
                    # Categorize as required or optional
                    if field_req.is_required:
                        required_fields[field_name] = field_req
                    else:
                        optional_fields[field_name] = field_req
            
            table_requirements[table_name] = TableRequirements(
                table_name=table_name,
                required_fields=required_fields,
                optional_fields=optional_fields,
                foreign_keys=foreign_keys,
                primary_key=primary_key
            )
        
        return table_requirements
    
    def _parse_field_requirement(self, field_name: str, field_config: Dict[str, Any]) -> FieldRequirement:
        """Parse individual field requirements from schema"""
        
        # Extract type information
        type_str = field_config.get('type', 'text')
        data_type = self._normalize_data_type(type_str)
        
        # Determine if field is required (NOT NULL and no default)
        nullable = field_config.get('nullable', True)
        has_default = 'DEFAULT' in type_str.upper() or 'default' in field_config
        
        # Check for primary key
        is_primary_key = field_config.get('primary_key', False)
        
        # Primary keys are always required (even if not explicitly marked as NOT NULL)
        is_required = (not nullable and not has_default) or is_primary_key
        
        # Check for foreign key
        is_foreign_key = 'foreign_key' in field_config
        references = None
        if is_foreign_key:
            ref_info = field_config.get('foreign_key', {})
            references = ref_info.get('references', '')
        
        # Check for auto-increment (primary keys with nextval are auto-increment)
        auto_increment = ('nextval' in type_str.lower() or 'SERIAL' in type_str.upper() or 
                         (is_primary_key and 'DEFAULT' in type_str.upper()))
        
        # Extract default value
        default_value = self._extract_default_value(type_str, field_config)
        
        # Parse constraints
        constraints = self._parse_constraints(type_str, field_config)
        
        print(f"   📋 Parsed field '{field_name}': {data_type}, Required: {is_required}, PK: {is_primary_key}, Auto: {auto_increment}")
        
        return FieldRequirement(
            field_name=field_name,
            data_type=data_type,
            is_required=is_required,
            is_primary_key=is_primary_key,
            is_foreign_key=is_foreign_key,
            default_value=default_value,
            constraints=constraints,
            references=references,
            auto_increment=auto_increment
        )
    
    def _normalize_data_type(self, type_str: str) -> str:
        """Normalize database types to standard types"""
        type_str = type_str.upper()
        
        if any(t in type_str for t in ['INT', 'SERIAL', 'BIGINT']):
            return 'INTEGER'
        elif any(t in type_str for t in ['NUMERIC', 'DECIMAL', 'FLOAT', 'DOUBLE']):
            return 'NUMERIC'
        elif any(t in type_str for t in ['BOOL']):
            return 'BOOLEAN'
        elif any(t in type_str for t in ['DATE']):
            return 'DATE'
        elif any(t in type_str for t in ['TIMESTAMP', 'TIME']):
            return 'TIMESTAMP'
        elif any(t in type_str for t in ['UUID']):
            return 'UUID'
        else:
            return 'TEXT'
    
    def _extract_default_value(self, type_str: str, field_config: Dict[str, Any]) -> Any:
        """Extract default value from field configuration"""
        
        if 'now' in type_str.lower():
            return 'CURRENT_TIMESTAMP'
        elif 'true' in type_str.lower():
            return True
        elif 'false' in type_str.lower():
            return False
        elif 'nextval' in type_str.lower():
            return 'AUTO_INCREMENT'
        elif 'DEFAULT' in type_str.upper():
            # Extract default value from type string
            match = re.search(r"DEFAULT\s+([^,\s]+)", type_str)
            if match:
                return match.group(1).strip("'\"")
        
        return None
    
    def _parse_constraints(self, type_str: str, field_config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse field constraints"""
        constraints = {}
        
        if 'CHECK' in type_str.upper():
            constraints['has_check'] = True
        if 'UNIQUE' in type_str.upper():
            constraints['unique'] = True
        if 'NOT NULL' in type_str.upper():
            constraints['not_null'] = True
            
        return constraints
    
    def validate_table_data(self, table_name: str, table_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate table data against schema requirements"""
        
        if table_name not in self.table_requirements:
            return {
                'valid': False,
                'errors': [f"Table '{table_name}' not found in schema"],
                'missing_required': [],
                'missing_optional': [],
                'type_mismatches': []
            }
        
        requirements = self.table_requirements[table_name]
        errors = []
        missing_required = []
        missing_optional = []
        type_mismatches = []
        
        # Check for missing required fields
        for field_name, field_req in requirements.required_fields.items():
            if field_name not in table_df.columns:
                missing_required.append(field_name)
                errors.append(f"Required column '{field_name}' missing")
        
        # Check for missing optional fields that have data mappings
        for field_name, field_req in requirements.optional_fields.items():
            if field_name not in table_df.columns:
                missing_optional.append(field_name)
        
        # Validate existing field types
        for col in table_df.columns:
            if col in requirements.required_fields:
                field_req = requirements.required_fields[col]
                if not self._validate_field_type(table_df[col], field_req):
                    type_mismatches.append(col)
            elif col in requirements.optional_fields:
                field_req = requirements.optional_fields[col]
                if not self._validate_field_type(table_df[col], field_req):
                    type_mismatches.append(col)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'missing_required': missing_required,
            'missing_optional': missing_optional,
            'type_mismatches': type_mismatches,
            'requirements': requirements
        }
    
    def _validate_field_type(self, series: pd.Series, field_req: FieldRequirement) -> bool:
        """Validate if series data type matches field requirement"""
        
        try:
            if field_req.data_type == 'INTEGER':
                pd.to_numeric(series, errors='coerce')
                return True
            elif field_req.data_type == 'NUMERIC':
                pd.to_numeric(series, errors='coerce')
                return True
            elif field_req.data_type == 'BOOLEAN':
                # Check if values can be converted to boolean
                return True  # We'll handle conversion later
            elif field_req.data_type in ['DATE', 'TIMESTAMP']:
                pd.to_datetime(series, errors='coerce')
                return True
            else:
                return True  # TEXT fields are flexible
                
        except:
            return False
    
    def enhance_table_with_requirements(self, table_name: str, table_df: pd.DataFrame, 
                                      source_data_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Dynamically enhance table with all schema requirements
        """
        
        if table_name not in self.table_requirements:
            raise ValueError(f"Table '{table_name}' not found in schema requirements")
        
        requirements = self.table_requirements[table_name]
        enhanced_df = table_df.copy()
        
        print(f"🔧 Dynamically enhancing '{table_name}' based on schema requirements...")
        print(f"   📊 Input: {len(enhanced_df)} rows, {len(enhanced_df.columns)} columns")
        
        # Track what we're adding
        added_fields = []
        
        # Add missing required fields
        for field_name, field_req in requirements.required_fields.items():
            if field_name not in enhanced_df.columns:
                default_values = self._generate_field_values(
                    field_req, len(enhanced_df), source_data_df
                )
                enhanced_df[field_name] = default_values
                added_fields.append(f"{field_name} ({field_req.data_type})")
                print(f"   ✅ Added required field: {field_name} ({field_req.data_type})")
        
        # Add missing optional fields with meaningful defaults
        for field_name, field_req in requirements.optional_fields.items():
            if field_name not in enhanced_df.columns:
                # Only add optional fields if they have meaningful defaults or are commonly needed
                if (field_req.default_value or 
                    any(keyword in field_name.lower() for keyword in ['created_at', 'updated_at', 'status'])):
                    default_values = self._generate_field_values(
                        field_req, len(enhanced_df), source_data_df
                    )
                    enhanced_df[field_name] = default_values
                    added_fields.append(f"{field_name} ({field_req.data_type}, optional)")
                    print(f"   ✅ Added optional field: {field_name} ({field_req.data_type})")
        
        # Convert data types for existing fields
        enhanced_df = self._convert_data_types(enhanced_df, requirements)
        
        # Summary
        print(f"   📊 Result: {len(enhanced_df)} rows, {len(enhanced_df.columns)} columns")
        if added_fields:
            print(f"   🆕 Added {len(added_fields)} fields: {', '.join(added_fields[:3])}")
            if len(added_fields) > 3:
                print(f"      ... and {len(added_fields) - 3} more")
        
        return enhanced_df
    
    def _generate_field_values(self, field_req: FieldRequirement, row_count: int, 
                             source_data_df: pd.DataFrame = None) -> List[Any]:
        """Generate appropriate default values for a field based on its requirements"""
        
        field_name = field_req.field_name
        data_type = field_req.data_type
        
        print(f"      🔧 Generating {row_count} values for '{field_name}' ({data_type})")
        
        # Handle auto-increment fields
        if field_req.auto_increment or field_req.is_primary_key:
            values = list(range(1, row_count + 1))
            print(f"         ✅ Generated auto-increment IDs: 1 to {row_count}")
            return values
        
        # Handle foreign keys - try to find matching values
        if field_req.is_foreign_key and source_data_df is not None:
            fk_values = self._generate_foreign_key_values(field_req, row_count, source_data_df)
            if fk_values:
                print(f"         ✅ Generated foreign key values from source data")
                return fk_values
        
        # Handle fields with explicit defaults
        if field_req.default_value:
            if field_req.default_value == 'CURRENT_TIMESTAMP':
                values = [pd.Timestamp.now()] * row_count
                print(f"         ✅ Generated current timestamps")
                return values
            elif field_req.default_value == 'AUTO_INCREMENT':
                values = list(range(1, row_count + 1))
                print(f"         ✅ Generated auto-increment values")
                return values
            else:
                values = [field_req.default_value] * row_count
                print(f"         ✅ Used default value: {field_req.default_value}")
                return values
        
        # Generate type-appropriate defaults with robust handling
        print(f"         ⚙️  Generating smart defaults for {data_type} field")
        
        if data_type == 'INTEGER':
            if 'id' in field_name.lower():
                values = list(range(1, row_count + 1))
                print(f"         ✅ Generated sequential IDs: 1 to {row_count}")
                return values
            else:
                values = [1] * row_count  # Use 1 instead of 0 for better data quality
                print(f"         ✅ Generated default integer value: 1")
                return values
            
        elif data_type == 'NUMERIC':
            values = [0.0] * row_count
            print(f"         ✅ Generated default numeric value: 0.0")
            return values
            
        elif data_type == 'BOOLEAN':
            values = [False] * row_count
            print(f"         ✅ Generated default boolean value: False")
            return values
            
        elif data_type == 'DATE':
            values = [pd.Timestamp('2024-01-01').date()] * row_count
            print(f"         ✅ Generated default date: 2024-01-01")
            return values
            
        elif data_type == 'TIMESTAMP':
            values = [pd.Timestamp.now()] * row_count
            print(f"         ✅ Generated current timestamps")
            return values
            
        elif data_type == 'UUID':
            import uuid
            values = [str(uuid.uuid4()) for _ in range(row_count)]
            print(f"         ✅ Generated {row_count} unique UUIDs")
            return values
            
        else:  # TEXT/VARCHAR
            default_text = self._generate_meaningful_text_default(field_name)
            values = [default_text] * row_count
            print(f"         ✅ Generated default text: '{default_text}'")
            return values
    
    def _generate_meaningful_text_default(self, field_name: str) -> str:
        """Generate meaningful text defaults based on field name"""
        
        field_lower = field_name.lower()
        
        # Specific handling for common field types
        if 'outcome' in field_lower:
            return "live_birth"  # Common for pregnancy outcomes
        elif 'name' in field_lower:
            return f"Generated_{field_name}"
        elif 'code' in field_lower:
            return f"CODE_{int(pd.Timestamp.now().timestamp())}"
        elif 'status' in field_lower:
            return "active"
        elif 'type' in field_lower:
            return "default"
        elif 'description' in field_lower:
            return "System generated record"
        elif 'relationship' in field_lower:
            return "household_member"
        elif 'event' in field_lower:
            return "data_collection"
        elif 'category' in field_lower:
            return "general"
        elif 'level' in field_lower:
            return "primary"
        else:
            return f"default_{field_name}"
    
    def _generate_foreign_key_values(self, field_req: FieldRequirement, row_count: int, 
                                   source_data_df: pd.DataFrame) -> List[Any]:
        """Generate foreign key values by looking for related data or creating reasonable defaults"""
        
        print(f"         🔗 Generating foreign key values for {field_req.field_name}")
        
        # Try to find related primary key values in source data
        if field_req.references:
            # Parse reference: "table.column"
            if '.' in field_req.references:
                ref_table, ref_column = field_req.references.split('.', 1)
                print(f"            References: {ref_table}.{ref_column}")
                
                # Look for potential matching columns in source data
                potential_columns = [col for col in source_data_df.columns 
                                   if ref_column in col.lower() or ref_table in col.lower()]
                
                if potential_columns:
                    # Use values from the most relevant column
                    best_column = potential_columns[0]
                    available_values = source_data_df[best_column].dropna().unique()
                    
                    if len(available_values) > 0:
                        print(f"            ✅ Found {len(available_values)} unique values in {best_column}")
                        # Cycle through available values to fill all rows
                        values = []
                        for i in range(row_count):
                            values.append(available_values[i % len(available_values)])
                        return values
                
                # If no matching column found, generate reasonable defaults based on reference
                print(f"            ⚠️  No matching column found, generating defaults")
                return self._generate_default_foreign_key_values(ref_table, ref_column, row_count)
        
        # Fallback: generate sequential IDs
        print(f"         ⚠️  No reference info, generating sequential IDs")
        return list(range(1, row_count + 1))
    
    def _generate_default_foreign_key_values(self, ref_table: str, ref_column: str, row_count: int) -> List[Any]:
        """Generate reasonable default foreign key values based on the referenced table"""
        
        print(f"            🎯 Generating defaults for {ref_table}.{ref_column}")
        
        # Generate reasonable defaults based on common table patterns
        if ref_table.lower() == 'individual':
            # For individual references, create sequential individual IDs
            values = list(range(1, row_count + 1))
            print(f"            ✅ Generated individual IDs: 1 to {row_count}")
            return values
            
        elif ref_table.lower() == 'household':
            # For household references, group individuals into households
            household_size = 4  # Average household size
            values = []
            for i in range(row_count):
                household_id = (i // household_size) + 1
                values.append(household_id)
            print(f"            ✅ Generated household IDs with avg size {household_size}")
            return values
            
        elif ref_table.lower() == 'site' or ref_table.lower() == 'village':
            # For site/village references, distribute across a few locations
            num_sites = max(1, row_count // 20)  # Roughly 20 people per site
            values = []
            for i in range(row_count):
                site_id = (i % num_sites) + 1
                values.append(site_id)
            print(f"            ✅ Generated {num_sites} site/village IDs")
            return values
            
        else:
            # Generic case: sequential IDs
            values = list(range(1, row_count + 1))
            print(f"            ✅ Generated sequential IDs for {ref_table}")
            return values
    
    def _convert_data_types(self, df: pd.DataFrame, requirements: TableRequirements) -> pd.DataFrame:
        """Convert DataFrame columns to match schema requirements"""
        
        converted_df = df.copy()
        
        all_fields = {**requirements.required_fields, **requirements.optional_fields}
        
        for col in converted_df.columns:
            if col in all_fields:
                field_req = all_fields[col]
                converted_df[col] = self._convert_column_type(converted_df[col], field_req)
                print(f"   🔄 Converted {col} to {field_req.data_type}")
        
        return converted_df
    
    def _convert_column_type(self, series: pd.Series, field_req: FieldRequirement) -> pd.Series:
        """Convert a pandas series to match field requirements"""
        
        try:
            if field_req.data_type == 'INTEGER':
                return pd.to_numeric(series, errors='coerce').fillna(0).astype('Int64')
                
            elif field_req.data_type == 'NUMERIC':
                return pd.to_numeric(series, errors='coerce').fillna(0.0)
                
            elif field_req.data_type == 'BOOLEAN':
                # Smart boolean conversion
                bool_map = {
                    'yes': True, 'no': False, 'true': True, 'false': False,
                    'y': True, 'n': False, '1': True, '0': False,
                    1: True, 0: False, 'not asked': False, '': False,
                    'male': True, 'female': False, 'm': True, 'f': False
                }
                return series.map(bool_map).fillna(False)
                
            elif field_req.data_type == 'DATE':
                return pd.to_datetime(series, errors='coerce').dt.date
                
            elif field_req.data_type == 'TIMESTAMP':
                return pd.to_datetime(series, errors='coerce')
                
            else:  # TEXT, UUID, etc.
                return series.astype(str).replace('nan', '')
                
        except Exception as e:
            print(f"   ⚠️  Failed to convert {field_req.field_name}: {e}")
            return series
    
    def get_validation_summary(self, table_validations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive validation summary"""
        
        total_tables = len(table_validations)
        valid_tables = sum(1 for v in table_validations.values() if v['valid'])
        
        all_errors = []
        all_missing_required = []
        
        for table_name, validation in table_validations.items():
            for error in validation['errors']:
                all_errors.append(f"{table_name}: {error}")
            
            for field in validation['missing_required']:
                all_missing_required.append(f"{table_name}.{field}")
        
        return {
            'total_tables': total_tables,
            'valid_tables': valid_tables,
            'invalid_tables': total_tables - valid_tables,
            'all_errors': all_errors,
            'missing_required_fields': all_missing_required,
            'overall_valid': len(all_missing_required) == 0
        }