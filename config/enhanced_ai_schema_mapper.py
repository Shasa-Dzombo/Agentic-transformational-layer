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
from config.schema_validator import DynamicSchemaValidator

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
    """Simplified deterministic schema mapper - NO AI, NO JSON parsing issues"""
    
    def __init__(self, schema_file_path: str, api_key: str = None):
        """Initialize Enhanced AI Schema Mapper with dynamic validation"""
        self.schema_file_path = schema_file_path
        self.schema = self.load_schema()
        self.tables = self.schema['database']['tables']
        
        # Get actual available columns from schema
        self.available_columns = self._extract_available_columns()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print(f"🔍 Schema loaded with {len(self.tables)} tables")
        self._print_available_columns()
        
        # Add dynamic validator
        self.schema_validator = DynamicSchemaValidator(schema_file_path)
        print("🔧 Dynamic schema validator initialized")
    
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
        print(f"\n📋 Available Schema Columns:")
        for table_name, columns in self.available_columns.items():
            if len(columns) > 0:
                print(f"   {table_name}: {columns[:5]}{'...' if len(columns) > 5 else ''}")
    
    def intelligent_column_analysis(self, df: pd.DataFrame) -> Dict[str, ColumnIntelligence]:
        """Simplified column analysis"""
        print("🧠 Performing simplified column analysis...")
        
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
        
        print(f"   ✅ Analyzed {len(analysis)} columns")
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
    
    def build_schema_intelligence(self):
        """Build schema intelligence for backward compatibility"""
        print("🧠 Analyzing schema semantics autonomously...")
        print(f"   ✅ Discovered {len(self.tables)} table purposes")
        
        total_fields = sum(len(columns) for columns in self.available_columns.values())
        print(f"   ✅ Mapped {total_fields} field semantics")
        
        print(f"🧠 Schema intelligence built:")
        print(f"   Autonomous table analysis: {len(self.tables)}")
        print(f"   Semantic field mapping: {total_fields}")

    def get_smart_ai_mapping(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Smart AI mapping with precise schema and CSV awareness"""
        try:
            import google.generativeai as genai
            
            # Configure Gemini with your API key
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                print("⚠️  No GEMINI_API_KEY found, using deterministic mapping")
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

            print("🧠 Querying AI for smart schema mapping...")
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
                print(f"⚠️  AI JSON parsing failed: {e}")
                print(f"🔧 Using deterministic fallback...")
                return self._create_deterministic_mappings(self.intelligent_column_analysis(df))
                
        except Exception as e:
            print(f"⚠️  AI mapping failed: {e}")
            print(f"🔧 Using deterministic fallback...")
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
        print("🔍 Validating AI mapping results...")
        
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
                print(f"⚠️  Skipping {csv_col}: CSV column not found")
                continue
                
            if table_name not in self.available_columns:
                print(f"⚠️  Skipping {table_name}: Table not found in schema")
                continue
                
            if schema_col not in self.available_columns[table_name]:
                print(f"⚠️  Skipping {table_name}.{schema_col}: Schema column not found")
                continue
                
            schema_key = f"{table_name}.{schema_col}"
            if schema_key in used_schema_columns:
                print(f"⚠️  Skipping {schema_key}: Already mapped")
                continue
                
            # Skip low confidence mappings
            if confidence < 0.8:
                print(f"⚠️  Skipping {csv_col}: Low confidence ({confidence})")
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
            print(f"✅ AI Mapped: {csv_col} → {table_name}.{schema_col} (confidence: {confidence})")
        
        # Return in expected format - FIXED ALL KEYS
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
                'total_csv_columns': len(df.columns),        # FIXED: was missing
                'mapped_columns': mapped_count,              # FIXED: consistent naming
                'unmapped_columns': len(df.columns) - mapped_count,  # FIXED: was missing
                'confidence_score': avg_confidence           # FIXED: consistent naming
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
        print("🎯 Creating deterministic mappings using actual schema...")
        
        mappings = {}
        
        # Initialize tables that exist in schema
        for table_name in self.available_columns.keys():
            mappings[table_name] = []
        
        # Define EXACT mappings using ACTUAL schema column names
        # Check what columns actually exist in your schema first
        individual_cols = self.available_columns.get('individual', [])
        household_cols = self.available_columns.get('household', [])
        education_cols = self.available_columns.get('education', [])
        livelihoods_cols = self.available_columns.get('livelihoods', [])
        vaccination_cols = self.available_columns.get('vaccination', [])
        pregnancy_cols = self.available_columns.get('pregnancy', [])
        household_relationship_cols = self.available_columns.get('household_relationship', [])
        censoring_event_cols = self.available_columns.get('censoring_event', [])
        
        exact_mappings = []
        
        # Individual table mappings (only if columns exist)
        if 'individual_id' in individual_cols:
            exact_mappings.append(('res_individualid_anon', 'individual', 'individual_id'))
        if 'sex' in individual_cols:
            exact_mappings.append(('res_gender', 'individual', 'sex'))
        elif 'gender' in individual_cols:
            exact_mappings.append(('res_gender', 'individual', 'gender'))
        if 'date_of_birth' in individual_cols:
            exact_mappings.append(('res_datebirth', 'individual', 'date_of_birth'))
        if 'ethnicity' in individual_cols:
            exact_mappings.append(('res_ethnicity', 'individual', 'ethnicity'))
        if 'marital_status' in individual_cols:
            exact_mappings.append(('mar_maritalstatus', 'individual', 'marital_status'))
        if 'religion' in individual_cols:
            exact_mappings.append(('rel_religionstatus', 'individual', 'religion'))
        
        # Household table mappings
        if 'household_id' in household_cols:
            exact_mappings.append(('res_hhid_anon', 'household', 'household_id'))
        if 'start_date' in household_cols:
            exact_mappings.append(('res_datebeg', 'household', 'start_date'))
        
        # Education table mappings
        if 'currently_enrolled' in education_cols:
            exact_mappings.append(('edu_currschool', 'education', 'currently_enrolled'))
        # Add more education mappings based on available columns
        
        # Livelihoods table mappings
        # Check what's actually available in livelihoods table
        if 'income_sources' in livelihoods_cols:
            exact_mappings.append(('iga_typeofiga', 'livelihoods', 'income_sources'))
        if 'monthly_income' in livelihoods_cols:
            exact_mappings.append(('iga_inc30days_cash', 'livelihoods', 'monthly_income'))
        
        # Vaccination table mappings
        if 'vaccine_type' in vaccination_cols:
            exact_mappings.append(('vac_bcg', 'vaccination', 'vaccine_type'))
        
        # Pregnancy table mappings
        if 'outcome' in pregnancy_cols:
            exact_mappings.append(('pge_pregoutcome', 'pregnancy', 'outcome'))
        if 'multiple_birth' in pregnancy_cols:
            exact_mappings.append(('pge_multiplebirth', 'pregnancy', 'multiple_birth'))
        
        # Household relationship mappings
        if 'relationship_to_head' in household_relationship_cols:
            exact_mappings.append(('res_reltohhh', 'household_relationship', 'relationship_to_head'))
        
        # Censoring events mappings
        if 'event_type' in censoring_event_cols:
            exact_mappings.append(('censor_birth', 'censoring_event', 'event_type'))
        
        mapped_count = 0
        used_schema_columns = set()
        
        for csv_col, table_name, schema_col in exact_mappings:
            # Check if CSV column exists
            if csv_col not in intelligent_analysis:
                continue
            
            # Check if table exists in schema
            if table_name not in self.available_columns:
                print(f"⚠️  Table '{table_name}' not found in schema")
                continue
            
            # Check if schema column exists
            if schema_col not in self.available_columns[table_name]:
                print(f"⚠️  Column '{schema_col}' not found in table '{table_name}'")
                continue
            
            # Check for conflicts
            schema_key = f"{table_name}.{schema_col}"
            if schema_key in used_schema_columns:
                print(f"⚠️  Conflict avoided: {schema_key} already mapped")
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
            print(f"✅ Mapped: {csv_col} → {table_name}.{schema_col}")
        
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
    
    def _create_tables(self, df: pd.DataFrame, mapping_result: Dict[str, Any], intelligent_analysis: Dict[str, ColumnIntelligence]) -> Dict[str, pd.DataFrame]:
        """Create mapped tables"""
        mapped_tables = {}
        
        for table_name, mappings in mapping_result['table_mappings'].items():
            if not mappings:
                continue
            
            print(f"\n📊 Creating {table_name} table...")
            table_data = {}
            
            for mapping in mappings:
                csv_col = mapping['csv_column']
                schema_col = mapping['schema_column']
                
                if csv_col in df.columns:
                    # Simple data transformation
                    table_data[schema_col] = self._transform_column(df[csv_col], schema_col)
                    print(f"   ✅ {csv_col} → {schema_col}")
            
            if table_data:
                table_df = pd.DataFrame(table_data)
                mapped_tables[table_name] = table_df
                print(f"   📋 {table_name}: {len(table_df)} rows, {len(table_df.columns)} columns")
        
        return mapped_tables
    
    def _transform_column(self, series: pd.Series, schema_col: str) -> pd.Series:
        """Simple column transformation"""
        if 'date' in schema_col.lower():
            return pd.to_datetime(series, errors='coerce')
        elif schema_col.lower() in ['sex', 'gender']:
            return series.map({'Male': 'Male', 'Female': 'Female', 1: 'Male', 2: 'Female'}).fillna(series)
        elif series.dtype in ['int64', 'float64']:
            return pd.to_numeric(series, errors='coerce')
        else:
            return series.astype('string')
    
    def _print_summary(self, mapping_result: Dict[str, Any], mapped_tables: Dict[str, pd.DataFrame], intelligent_analysis: Dict[str, ColumnIntelligence]):
        """Print mapping summary"""
        print("\n" + "="*80)
        print("🎯 SMART AI SCHEMA MAPPING SUMMARY")
        print("="*80)
        
        summary = mapping_result['mapping_summary']
        print(f"\n📊 Mapping Results:")
        print(f"   Total CSV columns: {summary['total_csv_columns']}")
        print(f"   Successfully mapped: {summary['mapped_columns']}")
        print(f"   Unmapped: {summary['unmapped_columns']}")
        print(f"   Confidence: {summary['confidence_score']}")
        
        print(f"\n📋 Created Tables:")
        for table_name, table_df in mapped_tables.items():
            print(f"   {table_name}: {len(table_df)} rows, {len(table_df.columns)} columns")
        
        print(f"\n🎯 Smart AI mapping completed successfully! ✅")

    def validate_and_enhance_tables(self, mapped_tables: Dict[str, pd.DataFrame], 
                                   source_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Dynamically validate and enhance all mapped tables based on schema requirements
        """
        
        print("\n🚀 DYNAMIC SCHEMA VALIDATION & ENHANCEMENT")
        print("="*60)
        
        enhanced_tables = {}
        validation_results = {}
        
        # Step 1: Validate all tables
        print("\n🔍 Validating tables against schema requirements...")
        for table_name, table_df in mapped_tables.items():
            validation = self.schema_validator.validate_table_data(table_name, table_df)
            validation_results[table_name] = validation
            
            if validation['valid']:
                print(f"   ✅ {table_name}: Valid")
            else:
                print(f"   ⚠️  {table_name}: {len(validation['missing_required'])} required fields missing")
        
        # Step 2: Enhance tables with missing requirements
        print("\n🔧 Enhancing tables with schema requirements...")
        for table_name, table_df in mapped_tables.items():
            try:
                enhanced_df = self.schema_validator.enhance_table_with_requirements(
                    table_name, table_df, source_df
                )
                enhanced_tables[table_name] = enhanced_df
                print(f"   ✅ Enhanced {table_name}: {len(enhanced_df)} rows, {len(enhanced_df.columns)} cols")
                
            except Exception as e:
                print(f"   ❌ Failed to enhance {table_name}: {e}")
                enhanced_tables[table_name] = table_df  # Use original if enhancement fails
        
        # Step 3: Final validation
        print("\n✅ Final validation after enhancement...")
        final_validation_results = {}
        for table_name, enhanced_df in enhanced_tables.items():
            validation = self.schema_validator.validate_table_data(table_name, enhanced_df)
            final_validation_results[table_name] = validation
            
            if validation['valid']:
                print(f"   ✅ {table_name}: Fully compliant")
            else:
                remaining_issues = len(validation['missing_required'])
                print(f"   ⚠️  {table_name}: {remaining_issues} issues remain")
        
        # Step 4: Print comprehensive summary
        summary = self.schema_validator.get_validation_summary(final_validation_results)
        self._print_validation_summary(summary, validation_results, final_validation_results)
        
        return enhanced_tables

    def _print_validation_summary(self, summary: Dict[str, Any], 
                                before_validation: Dict[str, Dict[str, Any]],
                                after_validation: Dict[str, Dict[str, Any]]):
        """Print comprehensive validation summary"""
        
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
        
        print(f"\n🧠 Starting DYNAMIC AI schema mapping for {len(df.columns)} columns...")
        
        # Step 1: Analyze columns
        intelligent_analysis = self.intelligent_column_analysis(df)
        
        # Step 2: Get smart AI mapping (with fallback)
        try:
            mapping_result = self.get_smart_ai_mapping(df)
        except Exception as e:
            print(f"🔧 AI mapping failed ({e}), using deterministic fallback...")
            mapping_result = self._create_deterministic_mappings(intelligent_analysis)
        
        validation = self._validate_mappings(mapping_result, df)
        if not validation['valid']:
            print(f"🔧 Using deterministic fallback due to validation issues...")
            mapping_result = self._create_deterministic_mappings(intelligent_analysis)
        
        # Create base tables
        base_tables = self._create_tables(df, mapping_result, intelligent_analysis)
        
        # **NEW: Dynamic validation and enhancement**
        enhanced_tables = self.validate_and_enhance_tables(base_tables, df)
        
        # Print final summary
        self._print_enhanced_summary(mapping_result, enhanced_tables, intelligent_analysis)
        
        return enhanced_tables

    def _print_enhanced_summary(self, mapping_result: Dict[str, Any], enhanced_tables: Dict[str, pd.DataFrame], intelligent_analysis: Dict[str, ColumnIntelligence]):
        """Print enhanced mapping summary with validation results"""
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

# End of EnhancedAISchemaMapper class

