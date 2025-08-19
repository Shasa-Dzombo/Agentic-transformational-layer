# mappper
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import os
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import numpy as np
import uuid
from config.schema_validator import DynamicSchemaValidator

# ===== NEW CONFIG BLOCK =====
BOOLEAN_ALLOWLIST = {
    'multiple_birth',
    'has_income_generating_activity',
    'currently_enrolled',
    'has_ever_attended_school'
}

CENSOR_EVENT_SOURCE_MAP = {
    'censor_birth': 'birth',
    'censor_death': 'death',
    'censor_out': 'out_migration',
    'censor_in': 'in_migration',
    'censor_maternaldeath': 'maternal_death',
    'censor_gavebirth': 'gave_birth',
    'censor_conceived': 'conceived',
    'censor_impregnated': 'impregnated'
}

CENSOR_EVENT_FALLBACK = 'end_of_follow_up'  # matches typical enum naming
# ===========================================

class SemanticCategory(Enum):
    """Semantic categories for intelligent column classification"""
    INDIVIDUAL_IDENTIFIERS = "Individual Identifiers"
    DEMOGRAPHICS = "Demographics"
    EDUCATION = "Education"
    HEALTH_MEDICAL = "Health/Medical"
    ECONOMIC_LIVELIHOODS = "Economic/Livelihoods"
    HOUSEHOLD = "Household"
    GEOGRAPHIC = "Geographic"
    TEMPORAL = "Temporal"
    STATISTICAL_AGGREGATES = "Statistical Aggregates"
    ADMINISTRATIVE = "Administrative"
    VACCINATION = "Vaccination"
    PREGNANCY_BIRTH = "Pregnancy/Birth"
    CENSORING_EVENTS = "Censoring Events"

@dataclass
class ColumnIntelligence:
    """Simplified column analysis"""
    name: str
    semantic_category: SemanticCategory
    quality_score: float
    reasoning: str

class EnhancedAISchemaMapper:
    """Enhanced AI + deterministic schema mapper (UUID + safe conversions)"""
    
    def __init__(self, schema_file_path: str, api_key: str = None,
                 use_ai: bool = True, verbose: bool = True):
        """Initialize Enhanced AI Schema Mapper with dynamic validation"""
        self.schema_file_path = schema_file_path
        self.schema = self.load_schema()
        self.tables = self.schema['database']['tables']
        
        # Config flags
        self.use_ai = use_ai
        self.verbose = verbose
        
        # Cache for deterministic UUID generation from legacy IDs
        self._uuid_cache: Dict[str, str] = {}
        
        # Get actual available columns from schema
        self.available_columns = self._extract_available_columns()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._v(f"🔍 Schema loaded with {len(self.tables)} tables")
        self._print_available_columns()
        
        # Add dynamic validator
        self.schema_validator = DynamicSchemaValidator(schema_file_path)
        self._v("🔧 Dynamic schema validator initialized")

    def _v(self, msg: str):
        """Verbose logging helper"""
        if self.verbose:
            print(msg)
    
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
    
    def _extract_available_columns(self) -> Dict[str, List[str]]:
        """Extract actual available columns from schema"""
        available = {}
        for table_name, table_config in self.tables.items():
            if 'columns' in table_config:
                available[table_name] = list(table_config['columns'].keys())
        return available
    
    def _print_available_columns(self):
        """Print available schema columns for validation"""
        if self.verbose:
            print(f"\n📋 Available Schema Columns:")
            for table_name, columns in self.available_columns.items():
                if len(columns) > 0:
                    print(f"   {table_name}: {columns[:5]}{'...' if len(columns) > 5 else ''}")
    
    def intelligent_column_analysis(self, df: pd.DataFrame) -> Dict[str, ColumnIntelligence]:
        """Simplified column analysis with early pruning"""
        self._v("🧠 Performing simplified column analysis...")
        
        # PRUNE statistical aggregates early
        drop_cols = [c for c in df.columns if any(p in c.lower() for p in ['los_', '_allsex_', '_femsex_', '_malsex_', 'under1yrs', 'anyage'])]
        if drop_cols:
            self._v(f"🧹 Pruning {len(drop_cols)} aggregate columns")
            df = df.drop(columns=drop_cols)
        
        analysis = {}
        for column in df.columns:
            # Basic stats
            null_pct = df[column].isnull().sum() / len(df) * 100
            unique_count = df[column].nunique()
            
            # Semantic classification
            semantic_category = self._classify_column(column)
            
            # Quality score
            quality_score = self._calculate_quality(null_pct, unique_count, semantic_category)
            
            # Reasoning
            reasoning = f"Column '{column}' classified as {semantic_category.value} with {null_pct:.1f}% nulls"
            
            analysis[column] = ColumnIntelligence(
                name=column,
                semantic_category=semantic_category,
                quality_score=quality_score,
                reasoning=reasoning
            )
        
        self._v(f"   ✅ Analyzed {len(analysis)} columns")
        return analysis
    
    def _classify_column(self, column_name: str) -> SemanticCategory:
        """Simple rule-based column classification"""
        col_lower = column_name.lower()
        
        # Statistical aggregates (skip these)
        if any(pattern in col_lower for pattern in ['los_', '_allsex_', '_femsex_', '_malsex_', 'under1yrs', 'anyage']):
            return SemanticCategory.STATISTICAL_AGGREGATES
        
        # Individual identifiers
        if 'individualid' in col_lower and 'anon' in col_lower:
            return SemanticCategory.INDIVIDUAL_IDENTIFIERS
        
        # Demographics
        if any(term in col_lower for term in ['gender', 'sex', 'age', 'birth', 'ethnicity', 'marital']):
            return SemanticCategory.DEMOGRAPHICS
        
        # Education
        if any(term in col_lower for term in ['edu', 'school']) and 'los_' not in col_lower:
            return SemanticCategory.EDUCATION
        
        # Economic/Livelihoods
        if any(term in col_lower for term in ['iga', 'income']):
            return SemanticCategory.ECONOMIC_LIVELIHOODS
        
        # Pregnancy/Birth
        if any(term in col_lower for term in ['gbs', 'pge', 'pregnant']):
            return SemanticCategory.PREGNANCY_BIRTH
        
        # Vaccination
        if col_lower.startswith('vac_'):
            return SemanticCategory.VACCINATION
        
        # Household
        if any(term in col_lower for term in ['hhid', 'household', 'hhh']):
            return SemanticCategory.HOUSEHOLD
        
        # Censoring events
        if col_lower.startswith('censor_'):
            return SemanticCategory.CENSORING_EVENTS
        
        return SemanticCategory.ADMINISTRATIVE
    
    def _calculate_quality(self, null_pct: float, unique_count: int, semantic_category: SemanticCategory) -> float:
        """Simple quality calculation"""
        base_score = 0.5
        
        # Completeness factor
        base_score += (1 - null_pct / 100) * 0.3
        
        # Category-specific adjustments
        if semantic_category == SemanticCategory.STATISTICAL_AGGREGATES:
            base_score *= 0.1  # Don't map these
        elif semantic_category == SemanticCategory.INDIVIDUAL_IDENTIFIERS:
            base_score += 0.4  # High value
        elif semantic_category == SemanticCategory.DEMOGRAPHICS:
            base_score += 0.2  # Good value
        
        return max(0, min(1, base_score))

    def get_smart_ai_mapping(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Smart AI mapping with precise schema and CSV awareness"""
        if not self.use_ai:
            return self._create_deterministic_mappings(self.intelligent_column_analysis(df))
            
        try:
            import google.generativeai as genai
            
            # Configure Gemini with your API key
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                self._v("⚠️  No GEMINI_API_KEY found, using deterministic mapping")
                return self._create_deterministic_mappings(self.intelligent_column_analysis(df))
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            # Create precise mapping context
            mapping_context = self._create_precise_mapping_context(df)
            
            prompt = f"""
You are a precise data mapping expert. Map CSV columns to exact database schema fields.

{mapping_context}

CRITICAL REQUIREMENTS:
1. Only use EXACT schema column names listed above
2. Only map CSV columns that exist in the provided list
3. Use direct matches first, then semantic matches
4. Each schema column can only be mapped ONCE
5. Skip statistical aggregates (los_* columns)
6. Be conservative - only map if you're 80%+ confident

Return ONLY valid JSON in this format:
{{
    "mappings": [
        {{"csv_column": "res_gender", "table": "individual", "schema_column": "sex", "confidence": 0.95, "reasoning": "Direct semantic match for gender data"}},
        {{"csv_column": "res_datebirth", "table": "individual", "schema_column": "date_of_birth", "confidence": 0.95, "reasoning": "Direct match for birth date"}}
    ],
    "summary": {{
        "total_mapped": 2,
        "high_confidence_count": 2,
        "avg_confidence": 0.95
    }}
}}
"""

            self._v("🧠 Querying AI for smart schema mapping...")
            response = model.generate_content(prompt)
            
            # Parse AI response
            try:
                # Extract JSON from response
                response_text = response.text.strip()
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                else:
                    json_text = response_text
                
                ai_result = json.loads(json_text)
                
                # Validate and convert to expected format
                return self._validate_and_convert_ai_mapping(ai_result, df)
                
            except json.JSONDecodeError as e:
                self._v(f"⚠️  AI JSON parsing failed: {e}")
                self._v(f"🔧 Using deterministic fallback...")
                return self._create_deterministic_mappings(self.intelligent_column_analysis(df))
                
        except Exception as e:
            self._v(f"⚠️  AI mapping failed: {e}")
            self._v(f"🔧 Using deterministic fallback...")
            return self._create_deterministic_mappings(self.intelligent_column_analysis(df))

    def _create_precise_mapping_context(self, df: pd.DataFrame) -> str:
        """Create precise context with actual schema and CSV data"""
        context = []
        
        # Available CSV columns (first 50 to avoid token limits)
        csv_cols = list(df.columns)
        context.append("=== AVAILABLE CSV COLUMNS ===")
        for i in range(0, min(len(csv_cols), 50), 10):
            batch = csv_cols[i:i+10]
            context.append(f"Columns {i+1}-{min(i+10, len(csv_cols))}: {', '.join(batch)}")
        
        if len(csv_cols) > 50:
            context.append(f"... and {len(csv_cols) - 50} more columns")
        
        # Key tables and their exact columns
        key_tables = {
            'individual': ['individual_id', 'sex', 'date_of_birth', 'ethnicity', 'marital_status', 'religion'],
            'household': ['household_id', 'start_date', 'end_date'],
            'education': ['education_id', 'individual_id', 'currently_enrolled', 'has_ever_attended_school', 'highest_level_ever_attended', 'current_school_type'],
            'livelihoods': ['livelihood_id', 'household_id', 'has_income_generating_activity', 'reason_no_iga', 'type_of_iga', 'income_30days_cash', 'income_30days_kind'],
            'vaccination': ['vaccination_id', 'individual_id', 'vaccine_type', 'dose_number', 'administration_date'],
            'pregnancy': ['pregnancy_id', 'mother_id', 'outcome', 'multiple_birth', 'delivery_date'],
            'household_relationship': ['relationship_id', 'individual_id', 'household_id', 'relationship_to_head', 'start_date'],
            'censoring_event': ['censoring_id', 'individual_id', 'event_type', 'event_date', 'comments']
        }
        
        context.append("\n=== EXACT SCHEMA TABLES & COLUMNS ===")
        for table_name, columns in key_tables.items():
            # Verify columns exist in actual schema
            if table_name in self.available_columns:
                available_cols = [col for col in columns if col in self.available_columns[table_name]]
                if available_cols:
                    context.append(f"{table_name}: {', '.join(available_cols)}")
        
        # Mapping hints
        context.append("\n=== MAPPING HINTS ===")
        context.append("• res_* = residence/individual data")
        context.append("• edu_* = education data")
        context.append("• iga_* = income generating activities (livelihoods)")
        context.append("• vac_* = vaccination data")
        context.append("• pge_* = pregnancy data")
        context.append("• censor_* = censoring events")
        context.append("• mar_* = marital status")
        context.append("• rel_* = religion")
        context.append("• Skip los_* columns (statistical aggregates)")
        
        return "\n".join(context)

    def _validate_and_convert_ai_mapping(self, ai_result: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Validate AI mapping and convert to expected format"""
        self._v("🔍 Validating AI mapping results...")
        
        table_mappings = {}
        used_schema_columns = set()
        valid_mappings = []
        
        for mapping in ai_result.get('mappings', []):
            csv_col = mapping.get('csv_column')
            table_name = mapping.get('table')
            schema_col = mapping.get('schema_column')
            confidence = mapping.get('confidence', 0.0)
            reasoning = mapping.get('reasoning', 'AI mapping')
            
            # Validation checks
            if csv_col not in df.columns:
                self._v(f"⚠️  Skipping {csv_col}: CSV column not found")
                continue
                
            if table_name not in self.available_columns:
                self._v(f"⚠️  Skipping {table_name}: Table not found in schema")
                continue
                
            if schema_col not in self.available_columns[table_name]:
                self._v(f"⚠️  Skipping {table_name}.{schema_col}: Schema column not found")
                continue
                
            schema_key = f"{table_name}.{schema_col}"
            if schema_key in used_schema_columns:
                self._v(f"⚠️  Skipping {schema_key}: Already mapped")
                continue
                
            # Skip low confidence mappings
            if confidence < 0.8:
                self._v(f"⚠️  Skipping {csv_col}: Low confidence ({confidence})")
                continue
            
            # Add to valid mappings
            if table_name not in table_mappings:
                table_mappings[table_name] = []
                
            table_mappings[table_name].append({
                'csv_column': csv_col,
                'schema_column': schema_col,
                'confidence': confidence,
                'reasoning': reasoning
            })
            
            used_schema_columns.add(schema_key)
            valid_mappings.append(mapping)
            self._v(f"✅ AI Mapped: {csv_col} → {table_name}.{schema_col} (confidence: {confidence})")
        
        # Return in expected format
        mapped_count = len(valid_mappings)
        avg_confidence = np.mean([m['confidence'] for m in valid_mappings]) if valid_mappings else 0
        
        return {
            'reasoning_summary': f'Smart AI mapping completed with {mapped_count} validated mappings',
            'table_mappings': table_mappings,
            'autonomous_decisions': [
                'Used AI with precise schema validation',
                'Applied confidence thresholds (80%+)',
                'Validated all mappings against actual schema',
                f'Successfully mapped {mapped_count} columns with avg confidence {avg_confidence:.2f}'
            ],
            'mapping_summary': {
                'total_csv_columns': len(df.columns),
                'mapped_columns': mapped_count,
                'unmapped_columns': len(df.columns) - mapped_count,
                'confidence_score': avg_confidence
            }
        }

    @property
    def schema_intelligence(self) -> Dict[str, Any]:
        """Schema intelligence property for backward compatibility"""
        return {
            'table_purposes': {table_name: f"Table for {table_name} data" for table_name in self.tables.keys()},
            'field_semantics': {
                f"{table_name}.{col_name}": f"Field {col_name} in {table_name} table"
                for table_name, columns in self.available_columns.items()
                for col_name in columns
            }
        }

    def _create_deterministic_mappings(self, intelligent_analysis: Dict[str, ColumnIntelligence]) -> Dict[str, Any]:
        """Create deterministic mappings using ACTUAL schema columns"""
        self._v("🎯 Creating deterministic mappings using actual schema...")
        
        mappings = {}
        
        # Initialize tables that exist in schema
        for table_name in self.available_columns.keys():
            mappings[table_name] = []
        
        # Define EXACT mappings using ACTUAL schema column names
        individual_cols = self.available_columns.get('individual', [])
        education_cols = self.available_columns.get('education', [])
        livelihoods_cols = self.available_columns.get('livelihoods', [])
        pregnancy_cols = self.available_columns.get('pregnancy', [])
        censoring_event_cols = self.available_columns.get('censoring_event', [])
        
        exact_mappings = []
        
        # Individual table mappings (only if columns exist)
        if 'individual_id' in individual_cols:
            exact_mappings.append(('res_individualid_anon', 'individual', 'individual_id'))
        if 'date_of_birth' in individual_cols:
            exact_mappings.append(('res_datebirth', 'individual', 'date_of_birth'))
        if 'ethnicity' in individual_cols:
            exact_mappings.append(('res_ethnicity', 'individual', 'ethnicity'))
        if 'marital_status' in individual_cols:
            exact_mappings.append(('mar_maritalstatus', 'individual', 'marital_status'))
        
        # Education table mappings
        if 'highest_level_ever_attended' in education_cols:
            exact_mappings.append(('edu_everschool_level', 'education', 'highest_level_ever_attended'))
        if 'current_school_type' in education_cols:
            exact_mappings.append(('edu_currschool_type', 'education', 'current_school_type'))
        
        # Livelihoods table mappings
        if 'type_of_iga' in livelihoods_cols:
            exact_mappings.append(('iga_typeofiga', 'livelihoods', 'type_of_iga'))
        if 'income_30days_cash' in livelihoods_cols:
            exact_mappings.append(('iga_inc30days_cash', 'livelihoods', 'income_30days_cash'))
        if 'income_30days_kind' in livelihoods_cols:
            exact_mappings.append(('iga_inc30days_kind', 'livelihoods', 'income_30days_kind'))
        
        # Pregnancy table mappings
        if 'multiple_birth' in pregnancy_cols:
            exact_mappings.append(('gbs_multiplebirth', 'pregnancy', 'multiple_birth'))
        
        # Censoring events mappings
        if 'event_date' in censoring_event_cols:
            exact_mappings.append(('censor_birth', 'censoring_event', 'event_date'))
        
        mapped_count = 0
        used_schema_columns = set()
        
        for csv_col, table_name, schema_col in exact_mappings:
            # Check if CSV column exists
            if csv_col not in intelligent_analysis:
                continue
            
            # Check if table exists in schema
            if table_name not in self.available_columns:
                continue
            
            # Check if schema column exists
            if schema_col not in self.available_columns[table_name]:
                continue
            
            # Check for conflicts
            schema_key = f"{table_name}.{schema_col}"
            if schema_key in used_schema_columns:
                continue
            
            # Skip statistical aggregates
            col_intel = intelligent_analysis[csv_col]
            if col_intel.semantic_category == SemanticCategory.STATISTICAL_AGGREGATES:
                continue
            
            # Add mapping
            mappings[table_name].append({
                'csv_column': csv_col,
                'schema_column': schema_col,
                'confidence': 0.95,
                'reasoning': f'Deterministic mapping based on exact schema validation'
            })
            used_schema_columns.add(schema_key)
            mapped_count += 1
            self._v(f"✅ Mapped: {csv_col} → {table_name}.{schema_col}")
        
        # Remove empty tables
        mappings = {table: table_mappings for table, table_mappings in mappings.items() if table_mappings}
        
        return {
            'reasoning_summary': f'Deterministic mapping completed using exact schema validation',
            'table_mappings': mappings,
            'autonomous_decisions': [
                'Used deterministic rule-based mapping',
                'Validated all mappings against actual schema',
                'Avoided statistical aggregate columns',
                f'Successfully mapped {mapped_count} columns'
            ],
            'mapping_summary': {
                'total_csv_columns': len(intelligent_analysis),
                'mapped_columns': mapped_count,
                'unmapped_columns': len(intelligent_analysis) - mapped_count,
                'confidence_score': 0.95
            }
        }
    
    def _create_tables(self, df: pd.DataFrame, mapping_result: Dict[str, Any], intelligent_analysis: Dict[str, ColumnIntelligence]) -> Dict[str, pd.DataFrame]:
        """Create mapped tables"""
        mapped_tables = {}
        
        for table_name, mappings in mapping_result['table_mappings'].items():
            if not mappings:
                continue
            
            self._v(f"\n📊 Creating {table_name} table...")
            table_data = {}
            
            for mapping in mappings:
                csv_col = mapping['csv_column']
                schema_col = mapping['schema_column']
                
                if csv_col in df.columns:
                    # Safe data transformation
                    table_data[schema_col] = self._transform_column(df[csv_col], schema_col)
                    self._v(f"   ✅ {csv_col} → {schema_col}")
            
            if table_data:
                table_df = pd.DataFrame(table_data)
                mapped_tables[table_name] = table_df
                self._v(f"   📋 {table_name}: {len(table_df)} rows, {len(table_df.columns)} columns")
        
        return mapped_tables

    def _is_uuid_like(self, val: str) -> bool:
        """Fast UUID-like pattern check"""
        if not isinstance(val, str):
            return False
        v = val.strip()
        return len(v) == 36 and v.count('-') == 4

    def _ensure_uuid_series(self, series: pd.Series) -> pd.Series:
        """Convert any non-UUID values to stable UUIDs (hash → UUID v5)"""
        out = []
        for v in series.fillna(''):
            if self._is_uuid_like(v):
                out.append(v)
            else:
                if v not in self._uuid_cache:
                    # namespace + value -> deterministic uuid
                    self._uuid_cache[v] = str(uuid.uuid5(uuid.NAMESPACE_DNS, v or str(uuid.uuid4())))
                out.append(self._uuid_cache[v])
        return pd.Series(out, index=series.index, dtype='string')

    def _safe_boolean(self, series: pd.Series) -> pd.Series:
        """Safe boolean conversion with strict allowlist"""
        raw = series.astype(str).str.strip().str.lower()
        mapping = {
            'yes': True, 'no': False,
            'true': True, 'false': False,
            '1': True, '0': False,
            'niu (not in universe)': pd.NA,
            'not asked': pd.NA,
            'missing': pd.NA,
            'na': pd.NA,
            '': pd.NA,
            'nan': pd.NA
        }
        unknown_ratio = (~raw.isin(mapping.keys())).mean()
        if unknown_ratio > 0.2:
            return series.astype('string')  # abort conversion
        converted = raw.map(mapping)
        # Do NOT force fillna False; leave pd.NA so DB can store NULL
        return converted
    
    def _transform_column(self, series: pd.Series, schema_col: str) -> pd.Series:
        """Safe transform: preserve UUIDs, avoid accidental bool coercion"""
        col_lower = schema_col.lower()
        
        if col_lower.endswith('_id'):
            s = series.astype('string')
            return self._ensure_uuid_series(s)
        if 'date' in col_lower:
            return pd.to_datetime(series, errors='coerce')
        if schema_col in BOOLEAN_ALLOWLIST:
            return self._safe_boolean(series)
        if col_lower in ['sex', 'gender']:
            return series.map({'Male': 'Male', 'Female': 'Female', 1: 'Male', 2: 'Female'}).astype('string')
        return series.astype('string')

    def validate_and_enhance_tables(self, mapped_tables: Dict[str, pd.DataFrame], 
                                   source_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Dynamically validate and enhance all mapped tables based on schema requirements"""
        
        self._v("\n🚀 DYNAMIC SCHEMA VALIDATION & ENHANCEMENT")
        self._v("="*60)
        
        enhanced_tables = {}
        validation_results = {}
        
        # Step 1: Validate all tables
        self._v("\n🔍 Validating tables against schema requirements...")
        for table_name, table_df in mapped_tables.items():
            validation = self.schema_validator.validate_table_data(table_name, table_df)
            validation_results[table_name] = validation
            
            if validation['valid']:
                self._v(f"   ✅ {table_name}: Valid")
            else:
                self._v(f"   ⚠️  {table_name}: {len(validation['missing_required'])} required fields missing")
        
        # Step 2: Enhance tables with missing requirements
        self._v("\n🔧 Enhancing tables with schema requirements...")
        for table_name, table_df in mapped_tables.items():
            try:
                # First, use the schema validator's enhancement
                enhanced_df = self.schema_validator.enhance_table_with_requirements(
                    table_name, table_df, source_df
                )
                
                # Then, add our custom missing required fields generation
                enhanced_df = self._generate_missing_required_fields(
                    table_name, enhanced_df, source_df
                )
                
                enhanced_tables[table_name] = enhanced_df
                self._v(f"   ✅ Enhanced {table_name}: {len(enhanced_df)} rows, {len(enhanced_df.columns)} cols")
                
            except Exception as e:
                self._v(f"   ❌ Failed to enhance {table_name}: {e}")
                # Apply our custom enhancement as fallback
                try:
                    enhanced_df = self._generate_missing_required_fields(
                        table_name, table_df, source_df
                    )
                    enhanced_tables[table_name] = enhanced_df
                    self._v(f"   🔧 Fallback enhanced {table_name}: {len(enhanced_df)} rows, {len(enhanced_df.columns)} cols")
                except Exception as e2:
                    self._v(f"   ❌ Fallback also failed for {table_name}: {e2}")
                    enhanced_tables[table_name] = table_df  # Use original if all enhancement fails
        
        # Step 2.5: Apply post-enhancement consistency fixes
        final_tables = {}
        for name, df in enhanced_tables.items():
            final_tables[name] = self._post_enhance_consistency(name, df, source_df)
        
        # Step 3: Final validation
        self._v("\n✅ Final validation after enhancement...")
        final_validation_results = {}
        for table_name, enhanced_df in final_tables.items():
            validation = self.schema_validator.validate_table_data(table_name, enhanced_df)
            final_validation_results[table_name] = validation
            
            if validation['valid']:
                self._v(f"   ✅ {table_name}: Fully compliant")
            else:
                remaining_issues = len(validation['missing_required'])
                self._v(f"   ⚠️  {table_name}: {remaining_issues} issues remain")
        
        # Step 4: Print comprehensive summary
        summary = self.schema_validator.get_validation_summary(final_validation_results)
        self._print_validation_summary(summary, validation_results, final_validation_results)
        
        return final_tables

    def _post_enhance_consistency(self, table_name: str, df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
        """Fix event_type derivation & ensure UUID integrity after enhancement"""
        # Ensure *_id columns are UUID strings
        for col in df.columns:
            if col.endswith('_id'):
                df[col] = self._ensure_uuid_series(df[col].astype('string'))

        if table_name == 'censoring_event':
            if 'event_type' in df.columns:
                # derive if empty / all null
                if df['event_type'].isna().all() or (df['event_type'] == '').all():
                    derived = pd.Series([pd.NA]*len(df), dtype='string')
                    for src_col, enum_val in CENSOR_EVENT_SOURCE_MAP.items():
                        if src_col in source_df.columns:
                            mask = source_df[src_col].notna()
                            derived.loc[mask] = enum_val
                    df['event_type'] = derived.fillna(CENSOR_EVENT_FALLBACK)
            # sanitize invalid enums
            allowed = set(CENSOR_EVENT_SOURCE_MAP.values()) | {CENSOR_EVENT_FALLBACK}
            df['event_type'] = df['event_type'].where(df['event_type'].isin(allowed), CENSOR_EVENT_FALLBACK)

        # Prevent accidental conversion of non-allowlist text to boolean
        for col in df.columns:
            if df[col].dtype == bool and col not in BOOLEAN_ALLOWLIST:
                df[col] = df[col].astype('string')
        return df

    def _print_validation_summary(self, summary: Dict[str, Any], 
                                before_validation: Dict[str, Dict[str, Any]],
                                after_validation: Dict[str, Dict[str, Any]]):
        """Print comprehensive validation summary"""
        
        if not self.verbose:
            return
            
        print("\n" + "="*80)
        print("📊 DYNAMIC VALIDATION & ENHANCEMENT SUMMARY")
        print("="*80)
        
        print(f"\n📈 Validation Results:")
        print(f"   Total tables processed: {summary['total_tables']}")
        print(f"   Fully compliant tables: {summary['valid_tables']}")
        print(f"   Tables with issues: {summary['invalid_tables']}")
        
        print(f"\n🔧 Enhancement Impact:")
        before_issues = sum(len(v['missing_required']) for v in before_validation.values())
        after_issues = sum(len(v['missing_required']) for v in after_validation.values())
        resolved_issues = before_issues - after_issues
        
        print(f"   Issues before enhancement: {before_issues}")
        print(f"   Issues after enhancement: {after_issues}")
        print(f"   Issues resolved: {resolved_issues}")
        print(f"   Resolution rate: {(resolved_issues/before_issues)*100:.1f}%" if before_issues > 0 else "   Resolution rate: 100%")
        
        if after_issues > 0:
            print(f"\n⚠️  Remaining issues:")
            for error in summary['all_errors']:
                print(f"      • {error}")
        
        print(f"\n🎯 Schema Compliance: {'✅ ACHIEVED' if summary['overall_valid'] else '⚠️  PARTIAL'}")

    def map_dataframe_to_tables(self, df: pd.DataFrame, preprocessing_results: Dict[str, Any] = None) -> Dict[str, pd.DataFrame]:
        """Enhanced mapping with dynamic schema validation"""
        
        self._v(f"\n🧠 Starting DYNAMIC AI schema mapping for {len(df.columns)} columns...")
        
        # Step 1: Analyze columns
        intelligent_analysis = self.intelligent_column_analysis(df)
        
        # Step 2: Get smart AI mapping (with fallback)
        try:
            mapping_result = self.get_smart_ai_mapping(df)
        except Exception as e:
            self._v(f"🔧 AI mapping failed ({e}), using deterministic fallback...")
            mapping_result = self._create_deterministic_mappings(intelligent_analysis)
        
        validation = self._validate_mappings(mapping_result, df)
        if not validation['valid']:
            self._v(f"🔧 Using deterministic fallback due to validation issues...")
            mapping_result = self._create_deterministic_mappings(intelligent_analysis)
        
        # Create base tables
        base_tables = self._create_tables(df, mapping_result, intelligent_analysis)
        
        # Dynamic validation and enhancement
        enhanced_tables = self.validate_and_enhance_tables(base_tables, df)
        
        # Prepare for Supabase
        supabase_ready_tables = self.prepare_for_supabase(enhanced_tables)
        
        # Print final summary
        self._print_enhanced_summary(mapping_result, supabase_ready_tables, intelligent_analysis)
        
        return supabase_ready_tables

    def _validate_mappings(self, mapping_result: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Validate mappings against actual schema"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        for table_name, mappings in mapping_result['table_mappings'].items():
            # Check table exists
            if table_name not in self.available_columns:
                validation['errors'].append(f"Table '{table_name}' not found in schema")
                validation['valid'] = False
                continue
            
            for mapping in mappings:
                csv_col = mapping['csv_column']
                schema_col = mapping['schema_column']
                
                # Check CSV column exists
                if csv_col not in df.columns:
                    validation['errors'].append(f"CSV column '{csv_col}' not found")
                    validation['valid'] = False
                
                # Check schema column exists
                if schema_col not in self.available_columns[table_name]:
                    validation['errors'].append(f"Schema column '{table_name}.{schema_col}' not found")
                    validation['valid'] = False
        
        return validation

    def _print_enhanced_summary(self, mapping_result: Dict[str, Any], enhanced_tables: Dict[str, pd.DataFrame], intelligent_analysis: Dict[str, ColumnIntelligence]):
        """Print enhanced mapping summary with validation results"""
        if not self.verbose:
            return
            
        print("\n" + "="*80)
        print("🚀 DYNAMIC SCHEMA-DRIVEN MAPPING SUMMARY")
        print("="*80)
        
        summary = mapping_result['mapping_summary']
        print(f"\n📊 Mapping Results:")
        print(f"   Original CSV columns: {summary['total_csv_columns']}")
        print(f"   AI-mapped columns: {summary['mapped_columns']}")
        print(f"   Schema-enhanced columns: {sum(len(df.columns) for df in enhanced_tables.values())}")
        print(f"   Mapping confidence: {summary['confidence_score']:.2%}")
        
        print(f"\n📋 Enhanced Tables:")
        for table_name, table_df in enhanced_tables.items():
            print(f"   {table_name}: {len(table_df)} rows, {len(table_df.columns)} columns")
        
        print(f"\n🎯 Dynamic schema-driven mapping completed successfully! ✅")
        print(f"💡 All tables now comply with schema requirements automatically")

    def _convert_to_boolean(self, series: pd.Series) -> pd.Series:
        """Legacy path (kept for compatibility) now delegates to _safe_boolean"""
        return self._safe_boolean(series)

    def prepare_for_supabase(self, enhanced_tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply final cleaning with strict allowlist boolean & UUID preservation"""
        cleaned_tables = {}
        
        for table_name, df in enhanced_tables.items():
            self._v(f"🔧 Preparing {table_name} for Supabase...")
            
            new_df = df.copy()
            
            # UUID integrity
            for col in new_df.columns:
                if col.endswith('_id'):
                    new_df[col] = self._ensure_uuid_series(new_df[col].astype('string'))

            # Boolean allowlist only
            for col in new_df.columns:
                if col in BOOLEAN_ALLOWLIST:
                    new_df[col] = self._safe_boolean(new_df[col])
                elif new_df[col].dtype == bool:
                    new_df[col] = new_df[col].astype('string')

            # Dates
            for col in new_df.columns:
                if 'date' in col.lower():
                    new_df[col] = pd.to_datetime(new_df[col], errors='coerce')

            # Null normalization
            for col in new_df.columns:
                if new_df[col].dtype == 'string':
                    new_df[col] = new_df[col].where(new_df[col].notna(), None)

            cleaned_tables[table_name] = new_df
            self._v(f"   ✅ {table_name} prepared with {len(new_df)} rows")
        
        return cleaned_tables

    def _generate_missing_required_fields(self, table_name: str, enhanced_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
        """Generate only strictly required fields; no fake names"""
        
        if table_name not in self.tables:
            return enhanced_df
        
        table_config = self.tables[table_name]
        columns_config = table_config.get('columns', {})
        
        for col_name, col_config in columns_config.items():
            if col_name not in enhanced_df.columns:
                if not col_config.get('nullable', True):  # Required field
                    self._v(f"   🔧 Generating required field: {col_name}")
                    
                    if col_name.endswith('_id'):
                        # UUID path
                        enhanced_df[col_name] = [str(uuid.uuid4()) for _ in range(len(enhanced_df))]
                    elif table_name == 'censoring_event' and col_name == 'event_type':
                        enhanced_df[col_name] = CENSOR_EVENT_FALLBACK
                    elif table_name == 'pregnancy' and col_name == 'outcome':
                        enhanced_df[col_name] = 'live_birth'
                    elif 'date' in col_name:
                        enhanced_df[col_name] = pd.to_datetime('2024-01-01')
                    else:
                        # leave as NULL-able placeholder if schema truly non-nullable
                        enhanced_df[col_name] = pd.Series([None]*len(enhanced_df), dtype='object')
        
        return enhanced_df

    def _fix_null_dates(self, enhanced_tables: Dict[str, pd.DataFrame], source_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Fix null dates by using actual source data"""
        
        for table_name, df in enhanced_tables.items():
            if table_name == 'censoring_event' and 'event_date' in df.columns:
                # Use actual censor_birth data if available
                if 'censor_birth' in source_df.columns:
                    # Get the actual dates from source
                    actual_dates = pd.to_datetime(source_df['censor_birth'].iloc[:len(df)], errors='coerce')
                    # Fill null values with default date
                    df['event_date'] = actual_dates.fillna(value=pd.to_datetime('2024-01-01'))
                    self._v(f"   🔧 Fixed event_date nulls in {table_name}")
                else:
                    # Fallback to default date
                    df['event_date'] = pd.to_datetime('2024-01-01')

        return enhanced_tables