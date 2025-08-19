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
        is_required = not nullable and not has_default
        
        # Check for primary key
        is_primary_key = field_config.get('primary_key', False)
        
        # Check for foreign key
        is_foreign_key = 'foreign_key' in field_config
        references = None
        if is_foreign_key:
            ref_info = field_config.get('foreign_key', {})
            references = ref_info.get('references', '')
        
        # Check for auto-increment
        auto_increment = 'nextval' in type_str or 'SERIAL' in type_str.upper()
        
        # Extract default value
        default_value = self._extract_default_value(type_str, field_config)
        
        # Parse constraints
        constraints = self._parse_constraints(type_str, field_config)
        
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
        
        # Add missing required fields
        for field_name, field_req in requirements.required_fields.items():
            if field_name not in enhanced_df.columns:
                default_values = self._generate_field_values(
                    field_req, len(enhanced_df), source_data_df
                )
                enhanced_df[field_name] = default_values
                print(f"   ✅ Added required field: {field_name} ({field_req.data_type})")
        
        # Add missing optional fields with defaults if they have meaningful defaults
        for field_name, field_req in requirements.optional_fields.items():
            if field_name not in enhanced_df.columns and field_req.default_value:
                default_values = self._generate_field_values(
                    field_req, len(enhanced_df), source_data_df
                )
                enhanced_df[field_name] = default_values
                print(f"   ✅ Added optional field: {field_name} ({field_req.data_type})")
        
        # Convert data types for existing fields
        enhanced_df = self._convert_data_types(enhanced_df, requirements)
        
        return enhanced_df
    
    def _generate_field_values(self, field_req: FieldRequirement, row_count: int, 
                             source_data_df: pd.DataFrame = None) -> List[Any]:
        """Generate appropriate default values for a field based on its requirements"""
        
        field_name = field_req.field_name
        data_type = field_req.data_type
        
        # Handle auto-increment fields
        if field_req.auto_increment or field_req.is_primary_key:
            return list(range(1, row_count + 1))
        
        # Handle foreign keys - try to find matching values
        if field_req.is_foreign_key and source_data_df is not None:
            return self._generate_foreign_key_values(field_req, row_count, source_data_df)
        
        # Handle fields with explicit defaults
        if field_req.default_value:
            if field_req.default_value == 'CURRENT_TIMESTAMP':
                return [pd.Timestamp.now()] * row_count
            elif field_req.default_value == 'AUTO_INCREMENT':
                return list(range(1, row_count + 1))
            else:
                return [field_req.default_value] * row_count
        
        # Generate type-appropriate defaults
        if data_type == 'INTEGER':
            if 'id' in field_name.lower():
                return list(range(1, row_count + 1))
            return [0] * row_count
            
        elif data_type == 'NUMERIC':
            return [0.0] * row_count
            
        elif data_type == 'BOOLEAN':
            # FIX 1: The default for an unknown boolean should be None (NULL), not False.
            return [None] * row_count
            
        elif data_type == 'DATE':
            return ['2024-01-01'] * row_count
            
        elif data_type == 'TIMESTAMP':
            return [pd.Timestamp.now()] * row_count
            
        elif data_type == 'UUID':
            import uuid
            return [str(uuid.uuid4()) for _ in range(row_count)]
            
        else:  # TEXT
            return [self._generate_meaningful_text_default(field_name)] * row_count
    
    def _generate_meaningful_text_default(self, field_name: str) -> str:
        """Generate meaningful text defaults based on field name"""
        
        field_lower = field_name.lower()
        
        if 'name' in field_lower:
            return f"Generated_{field_name}"
        elif 'code' in field_lower:
            return f"CODE_{int(pd.Timestamp.now().timestamp())}"
        elif 'status' in field_lower:
            return "active"
        elif 'type' in field_lower:
            return "default"
        elif 'outcome' in field_lower:
            return "unknown"
        elif 'description' in field_lower:
            return "System generated"
        else:
            return f"default_{field_name}"
    
    def _generate_foreign_key_values(self, field_req: FieldRequirement, row_count: int, 
                                   source_data_df: pd.DataFrame) -> List[Any]:
        """
        Generate foreign key values by looking for related data, now with
        correct data type handling for UUIDs.
        """
        if not field_req.references or '.' not in field_req.references:
            # Fallback if reference info is missing
            return list(range(1, row_count + 1))

        ref_table, ref_column = field_req.references.split('.', 1)

        # Find the schema requirements for the referenced primary key to get its type
        ref_table_reqs = self.table_requirements.get(ref_table)
        ref_field_req = ref_table_reqs.required_fields.get(ref_column) or ref_table_reqs.optional_fields.get(ref_column) if ref_table_reqs else None

        if not ref_field_req:
            # Fallback if referenced column schema can't be found
            return list(range(1, row_count + 1))

        # Determine the expected data type (e.g., 'UUID' or 'INTEGER')
        expected_type = ref_field_req.data_type

        # Look for potential matching columns in the original source data
        potential_columns = [col for col in source_data_df.columns 
                           if ref_column in col.lower() or ref_table in col.lower()]
        
        if potential_columns:
            source_column = potential_columns[0]
            ref_values = source_data_df[source_column]

            # THIS IS THE FIX: Only convert to integer if the referenced key is an integer.
            # Otherwise, treat it as a string (for UUIDs).
            if expected_type == 'INTEGER':
                # Safely convert to numeric, then to nullable integer
                return pd.to_numeric(ref_values, errors='coerce').fillna(0).astype(int).tolist()[:row_count]
            else: # For UUID, TEXT, etc.
                # Convert to string and handle potential nulls
                return ref_values.astype(str).fillna('').tolist()[:row_count]

        # Final fallback if no matching source column is found
        return list(range(1, row_count + 1))

    def _convert_data_types(self, df: pd.DataFrame, requirements: TableRequirements) -> pd.DataFrame:
        """Convert DataFrame columns to match schema requirements"""
        
        converted_df = df.copy()
        
        all_fields = {**requirements.required_fields, **requirements.optional_fields}
        
        for col in converted_df.columns:
            if col in all_fields:
                field_req = all_fields[col]
                # This is where the conversion happens. The fix is in the method below.
                converted_df[col] = self._convert_column_type(converted_df[col], field_req)
                print(f"   🔄 Converted {col} to {field_req.data_type}")
        
        return converted_df
    
    def _convert_column_type(self, series: pd.Series, field_req: FieldRequirement) -> pd.Series:
        """
        Convert a pandas series to match schema requirements, correctly handling
        UUIDs, integers, and other types gracefully.
        """
        expected_type = field_req.data_type
        field_name = field_req.field_name

        try:
            # If the schema explicitly expects a UUID, treat it as a string.
            # This is the primary fix: respect the schema's intent for UUIDs.
            if expected_type == 'UUID':
                # Ensure all values are strings and replace any 'nan' strings from previous steps.
                return series.astype(str).replace('nan', '').fillna('')

            # If the schema expects an INTEGER, convert to numeric.
            if expected_type == 'INTEGER':
                # Coerce errors will turn non-numeric values (like UUIDs) into NaT.
                converted = pd.to_numeric(series, errors='coerce')
                # Fill any resulting nulls with 0 and cast to a nullable integer type.
                return converted.fillna(0).astype('Int64')

            elif expected_type == 'NUMERIC':
                converted = pd.to_numeric(series, errors='coerce')
                return converted.fillna(0.0)

            elif expected_type == 'BOOLEAN':
                return self._convert_to_boolean_safe(series)
                
            elif expected_type in ['DATE', 'TIMESTAMP']:
                converted = pd.to_datetime(series, errors='coerce')
                # pd.NaT is the correct null representation for datetime types.
                return converted.fillna(pd.NaT)
                
            else:  # Default to TEXT for any other type
                return series.astype(str).fillna('')
            
        except Exception as e:
            print(f"   ⚠️  Type conversion failed for '{field_name}' to '{expected_type}': {e}. Falling back to string.")
            # Safe fallback: if any conversion fails, convert to string to prevent crashes.
            return series.astype(str).fillna('')
    
    def _convert_to_boolean_safe(self, series: pd.Series) -> pd.Series:
        """Safe boolean conversion handling survey responses"""
        
        try:
            # Create boolean mapping for survey responses
            bool_mapping = {
                # Standard boolean values
                'yes': True, 'no': False, 'true': True, 'false': False,
                '1': True, '0': False, 1: True, 0: False, 1.0: True, 0.0: False,
                
                # Survey-specific responses that should correctly map to None (NULL)
                'niu (not in universe)': None,
                'not asked': None,
                "don't know": None,
                'missing': None,
                'na': None,
                '': None,
                'nan': None
            }
            
            # Convert to string and lowercase for robust mapping
            # Using .get allows for a default value (None) if a key is not in the map
            str_series = series.astype(str).str.lower().str.strip()
            result = str_series.map(bool_mapping)

            # FIX 2: Ensure that any value NOT in the map becomes None.
            # This prevents pandas from trying to fill with a default that overwrites our explicit `None` mappings.
            # The result is a series containing only True, False, or None.
            return result.where(pd.notna(result), None).astype('object')
            
        except Exception as e:
            print(f"   ⚠️  Boolean conversion error: {e}")
            # Ensure the fallback is a series of None values with the correct object type
            return pd.Series([None] * len(series), dtype='object')
    
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
    
    def _is_uuid_format(self, value: str) -> bool:
        """Check if a string value looks like a UUID"""
        if not isinstance(value, str) or len(value) != 36:
            return False
        
        # UUID format: 8-4-4-4-12 characters with hyphens
        uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        import re
        return bool(re.match(uuid_pattern, value))

    def _detect_actual_data_type(self, series: pd.Series) -> str:
        """Detect the actual data type from the series content"""
        if len(series) == 0:
            return 'TEXT'
        
        # Get first non-null value
        sample_value = None
        for val in series.dropna():
            if pd.notna(val):
                sample_value = str(val)
                break
        
        if not sample_value:
            return 'TEXT'
        
        # Check for UUID
        if self._is_uuid_format(sample_value):
            return 'UUID'
        
        # Check for numeric
        try:
            float(sample_value)
            return 'NUMERIC'
        except:
            pass
        
        # Check for boolean-like
        if sample_value.lower() in ['true', 'false', '1', '0', 'yes', 'no']:
            return 'BOOLEAN'
        
        return 'TEXT'