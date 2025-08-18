#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Filter incoming CSV columns to match HDSS schema fields using hybrid approach.
- Primary: Fuzzy + deterministic matching (fast)
- Secondary: AI refinement for matched columns only (selective)
- Prevents duplicate field mappings (one-to-one mapping)
- Outputs only the filtered DataFrame with schema field names

Usage:
  python filter_hdss_csv.py --input /path/to/incoming.csv --output ./filtered_data.csv
"""
import os
import re
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict

import pandas as pd

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    pass

# Try rapidfuzz for better fuzzy; fall back to difflib
try:
    from rapidfuzz import fuzz
    def fuzzy_ratio(a: str, b: str) -> float:
        return fuzz.token_set_ratio(a, b) / 100.0
except Exception:
    from difflib import SequenceMatcher
    def _tok_sort(s: str) -> str:
        toks = re.split(r"[_\s]+", s)
        toks = sorted([t for t in toks if t])
        return "_".join(toks)
    def fuzzy_ratio(a: str, b: str) -> float:
        r1 = SequenceMatcher(None, a, b).ratio()
        r2 = SequenceMatcher(None, _tok_sort(a), _tok_sort(b)).ratio()
        return max(r1, r2)

# =========================
# 1) HDSS SCHEMA DEFINITION
# =========================
SCHEMA: Dict[str, List[str]] = {
    # 1. Country and Site Structure
    "Country": ["country_id", "name", "code", "status"],
    "Site": ["site_id", "country_id", "name", "description", "location_name",
             "longitude", "latitude", "address", "start_date", "end_date"],

    # 2. Geographic and Dwelling Hierarchy
    "Village": ["village_id", "site_id", "name"],
    "Structure": ["structure_id", "village_id", "structure_code", "address_description", "structure_name"],
    "DwellingUnit": ["dwelling_unit_id", "structure_id", "unit_code", "unit_name"],

    # 3. Household and Individual
    "Household": ["household_id", "dwelling_unit_id", "household_code", "head_individual_id", "start_date", "end_date"],
    "Individual": ["individual_id", "household_id", "first_name", "last_name", "sex",
                   "date_of_birth", "national_id", "is_alive", "date_of_death", "religion",
                   "ethnicity", "marital_status"],
    "HouseholdRelationship": ["relationship_id", "individual_id", "household_id",
                              "relationship_to_head", "start_date", "end_date"],
    "ResidencyEpisode": ["episode_id", "individual_id", "household_id", "start_date", "event_type"],

    # 4. Demographic and Health Events
    "BirthEvent": ["birth_id", "individual_id", "mother_id", "birth_date", "birth_location", "attendant_type"],
    "DeathEvent": ["death_id", "individual_id", "death_date", "cause_of_death", "verbal_autopsy_done"],
    "MigrationEvent": ["migration_id", "individual_id", "migration_type", "from_location",
                       "to_location", "migration_date", "reason"],
    "CensoringEvent": ["censoring_id", "individual_id", "event_type", "event_date", "comments"],
    "Vaccination": ["vaccination_id", "individual_id", "age_at_vaccination_months", "vaccine_type",
                    "dose_number", "administration_date"],
    "Pregnancy": ["pregnancy_id", "mother_id", "outcome", "delivery_date", "birth_weight",
                  "multiple_birth", "preterm", "full_term", "neonatal_death",
                  "maternal_complication", "c_section_type"],

    # 5. Education, Amenities, Livelihood
    "Education": ["education_id", "individual_id", "school_level", "currently_enrolled",
                  "transition_status", "has_ever_attended_school", "highest_level_ever_attended",
                  "highest_class_completed", "current_school_type"],
    "HouseholdAmenities": ["amenity_id", "household_id", "water_source", "toilet_type",
                           "electricity", "waste_disposa"],
    "Livelihoods": ["livelihood_id", "household_id", "income_sources", "monthly_income",
                    "expenditure", "shocks_experienced", "has_income_generating_activity",
                    "reason_no_iga", "type_of_iga", "income_30days_cash", "income_30days_kind"],
}

# Deterministic aliases
ALIASES: Dict[str, str] = {
    "fname": "first_name", "first": "first_name", "first-name": "first_name",
    "surname": "last_name", "lname": "last_name", "last": "last_name", "last-name": "last_name",
    "gender": "sex", "gender_cd": "sex", "sex_cd": "sex",
    "dob": "date_of_birth", "birthdate": "date_of_birth", "date-of-birth": "date_of_birth",
    "nid": "national_id", "nat_id": "national_id", "nationalid": "national_id", "id_number": "national_id",
    "alive": "is_alive", "living": "is_alive", "isalive": "is_alive",
    "dod": "date_of_death", "deathdate": "date_of_death",
    "village": "name", "structurecode": "structure_code", "structure-name": "structure_name",
    "unitcode": "unit_code", "unit-name": "unit_name",
    "hh_code": "household_code", "householdcode": "household_code",
    "head_id": "head_individual_id",
    "birth_location_name": "birth_location", "attendant": "attendant_type",
    "cod": "cause_of_death", "cause": "cause_of_death", "va_done": "verbal_autopsy_done",
    "mig_type": "migration_type", "move_type": "migration_type",
    "from": "from_location", "to": "to_location", "mig_date": "migration_date",
    "age_months": "age_at_vaccination_months", "age_at_vaccination": "age_at_vaccination_months",
    "dose": "dose_number", "dose_no": "dose_number", "admin_date": "administration_date",
    "enrolled": "currently_enrolled", "transition": "transition_status",
    "highest_level": "highest_level_ever_attended", "highest_class": "highest_class_completed",
    "school_type": "current_school_type", "waste_disposal": "waste_disposa",
    "iga": "has_income_generating_activity", "reason_no_business": "reason_no_iga",
    "type_iga": "type_of_iga", "lat": "latitude", "lon": "longitude", "long": "longitude",
    "site_name": "name", "country_name": "name", "id": "individual_id",
    
    # Additional domain-specific aliases for HDSS
    "individualid": "individual_id", "individual_id_anon": "individual_id",
    "hhid": "household_id", "household_id_anon": "household_id", "hh_id": "household_id",
    "locationid": "dwelling_unit_id", "location_id": "dwelling_unit_id",
    "observationid": "episode_id", "observation_id": "episode_id",
    "pregid": "pregnancy_id", "preg_id": "pregnancy_id",
    "birthid": "birth_id", "birth_id": "birth_id",
    "datebirth": "date_of_birth", "date_birth": "date_of_birth",
    "datebeg": "start_date", "date_beg": "start_date",
    "eventdate": "event_date", "event_date": "event_date",
    "multiplebirth": "multiple_birth", "multiple_birth": "multiple_birth",
    "maritalstatus": "marital_status", "marital_status": "marital_status",
    "hasiga": "has_income_generating_activity", "has_iga": "has_income_generating_activity",
    "typeofiga": "type_of_iga", "type_iga": "type_of_iga",
    "inc30days_cash": "income_30days_cash", "income_30days_cash": "income_30days_cash",
    "inc30days_kind": "income_30days_kind", "income_30days_kind": "income_30days_kind",
    "everschool": "has_ever_attended_school", "ever_school": "has_ever_attended_school",
    "currschool": "currently_enrolled", "curr_school": "currently_enrolled",
    "reltohhh": "relationship_to_head", "rel_to_hhh": "relationship_to_head",
    "religionstatus": "religion", "religion_status": "religion",
}

# Build reverse lookup helpers
ALL_FIELDS = set()
FIELD_TO_TABLE: Dict[str, List[str]] = {}
for tbl, cols in SCHEMA.items():
    for c in cols:
        ALL_FIELDS.add(c)
        FIELD_TO_TABLE.setdefault(c, []).append(tbl)

def normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"[/\-:]+", "_", s)
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def fast_match(col: str, available_fields: set, threshold: float = 0.50) -> Tuple[str, str, float]:
    """
    Fast deterministic + fuzzy matching with higher accuracy thresholds
    """
    ncol = normalize(col)
    
    # 1) Exact match
    if ncol in available_fields:
        return ncol, "exact", 1.0
    
    # 2) Alias match
    if ncol in ALIASES:
        tgt = ALIASES[ncol]
        if tgt in available_fields:
            return tgt, "alias", 0.97
    
    # 3) Fuzzy match with much higher thresholds for accuracy
    best_field, best_score = "", 0.0
    for field in available_fields:
        score = fuzzy_ratio(ncol, field)
        if score > best_score:
            best_score = score
            best_field = field
    
    # Much stricter thresholds for better accuracy
    if best_score >= 0.80:  # Very high confidence - accept directly
        return best_field, "fuzzy_high", float(best_score)
    elif best_score >= threshold:  # Medium-high confidence - potential AI candidate
        return best_field, "fuzzy", float(best_score)
    
    return "", "none", 0.0

def should_refine_with_ai(col: str, field: str, score: float) -> bool:
    """
    Smarter filtering for AI refinement - focus on semantic similarity
    """
    ncol = normalize(col)
    nfield = normalize(field)
    
    # Skip very short or uninformative column names
    if len(ncol) <= 3:
        return False
    
    # Skip columns that are clearly computed/derived fields
    skip_patterns = [
        '_all', '_fem', '_mal', 'under', 'plus', 'to', 'yrs', 'sex_',
        '_anyage', '_10to', '_15to', '_20to', '_25to', '_30to', '_35to', '_40to', '_45to', '_50to', '_55to', '_60plus',
        'chddth_', 'pardth_', 'hhhdth_', 'dth_'
    ]
    if any(pattern in ncol for pattern in skip_patterns):
        return False
    
    # Only refine if score is in the "uncertain but promising" range
    if score < 0.50 or score > 0.70:  # Narrower range for AI consideration
        return False
    
    # Enhanced semantic checks - require some meaningful word overlap
    col_words = set(ncol.split('_'))
    field_words = set(nfield.split('_'))
    common_words = col_words & field_words
    
    # Must have at least one meaningful common word (not just short connectors)
    meaningful_common = [word for word in common_words if len(word) > 2]
    if len(meaningful_common) == 0:
        return False
    
    # Check for obvious mismatches by data type/domain
    data_type_mismatches = [
        ('gender', ['date', 'name', 'id']),
        ('date', ['gender', 'sex', 'name', 'type']),
        ('id', ['date', 'gender', 'sex', 'name']),
        ('name', ['date', 'gender', 'sex', 'id']),
        ('age', ['name', 'id', 'type']),
        ('type', ['date', 'age', 'id'])
    ]
    
    for col_pattern, incompatible_patterns in data_type_mismatches:
        if col_pattern in ncol:
            if any(pattern in nfield for pattern in incompatible_patterns):
                return False
    
    return True

def gemini_refine_match(col: str, fuzzy_match: str, available_fields: set, api_key: str, model: str = "gemini-2.5-flash") -> Tuple[str, str, float]:
    """
    Enhanced Gemini AI refinement with better context and validation
    """
    if not api_key:
        return "", "no_ai", 0.0
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        
        # Get semantically similar candidates for better AI context
        ncol = normalize(col)
        candidates = []
        
        for field in available_fields:
            score = fuzzy_ratio(ncol, field)
            if score >= 0.45:  # Lower threshold for candidates
                candidates.append((field, score))
        
        # Sort and take top candidates
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:8]
        candidate_fields = [c[0] for c in candidates]
        
        # Enhanced prompt with semantic context
        prompt = (
            f"You are mapping a CSV column to HDSS (Health Demographics) schema fields.\n\n"
            f"CSV Column: '{col}'\n"
            f"Current fuzzy match: '{fuzzy_match}'\n"
            f"Alternative candidates: {candidate_fields}\n\n"
            f"Context clues:\n"
            f"- 'res_' = residence/demographic data\n"
            f"- 'edu_' = education data\n"
            f"- 'iga_' = income generating activity\n"
            f"- 'pge_' = pregnancy episode\n"
            f"- 'gbs_' = general birth survey\n"
            f"- 'mar_' = marital status\n"
            f"- 'vac_' = vaccination\n"
            f"- 'censor_' = censoring events\n\n"
            f"Return ONLY:\n"
            f"- The EXACT field name if you find a better semantic match\n"
            f"- '{fuzzy_match}' if the current match is reasonable\n"
            f"- 'NONE' if no good match exists\n\n"
            f"Answer:"
        )
        
        response = model_instance.generate_content(prompt)
        result = (response.text or "").strip()
        result = normalize(result.split('\n')[0])
        
        if result == 'none':
            return "", "ai_rejected", 0.0
        
        if result == normalize(fuzzy_match):
            return fuzzy_match, "ai_confirmed", 0.80
        
        if result in available_fields:
            return result, "ai_refined", 0.85
        
        # AI returned something not available, reject the match
        return "", "ai_invalid", 0.0
            
    except Exception as e:
        print(f"AI refinement failed for '{col}': {e}")
        return "", "ai_failed", 0.0

# Update main function parameters for higher accuracy:
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV file path")
    ap.add_argument("--output", default="./filtered_data.csv", help="Output filtered DataFrame")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.65, help="Fuzzy matching threshold (higher = more accurate)")
    ap.add_argument("--ai-refinement-threshold", type=float, default=0.80, help="Score threshold for AI refinement")
    ap.add_argument("--max-ai-calls", type=int, default=25, help="Maximum AI calls to make")
    ap.add_argument("--use-ai-refinement", type=str, default="true", choices=["true","false"], help="Use AI for uncertain fuzzy matches")
    ap.add_argument("--ai-model", type=str, default="gemini-2.5-flash")
    ap.add_argument("--ai-api-key", type=str, default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY") or "")
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded data: {len(df)} rows × {len(df.columns)} columns")

    # Phase 1: Fast matching with smarter thresholds
    print("Phase 1: Fast matching (exact/alias/fuzzy)...")
    
    mapped_columns = {}
    available_fields = ALL_FIELDS.copy()
    fast_matches = []  # Store fuzzy matches for potential AI refinement
    table_summary = defaultdict(list)
    
    # Sort columns by importance
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    other_cols = [col for col in df.columns if col not in id_cols]
    sorted_columns = id_cols + other_cols
    
    for col in sorted_columns:
        field, match_type, score = fast_match(col, available_fields, args.fuzzy_threshold)
        
        if field:
            if match_type == "fuzzy" and score < args.ai_refinement_threshold:
                # Smart filtering for AI refinement candidates
                if should_refine_with_ai(col, field, score):
                    fast_matches.append((col, field, match_type, score))
                else:
                    # Accept the fuzzy match as-is (not worth AI refinement)
                    mapped_columns[col] = field
                    available_fields.remove(field)
                    
                    tables = FIELD_TO_TABLE.get(field, ["Unknown"])
                    for table in tables:
                        table_summary[table].append({
                            "source_column": col,
                            "schema_field": field,
                            "match_type": "fuzzy_accepted",
                            "score": score
                        })
            else:
                # Direct mapping for exact/alias/high-confidence fuzzy
                mapped_columns[col] = field
                available_fields.remove(field)
                
                tables = FIELD_TO_TABLE.get(field, ["Unknown"])
                for table in tables:
                    table_summary[table].append({
                        "source_column": col,
                        "schema_field": field,
                        "match_type": match_type,
                        "score": score
                    })

    # Sort AI candidates by score (best first) and limit the number
    fast_matches = sorted(fast_matches, key=lambda x: x[3], reverse=True)
    fast_matches = fast_matches[:args.max_ai_calls]
    
    print(f"Fast matching results: {len(mapped_columns)} direct matches, {len(fast_matches)} selected for AI refinement")

    # Phase 2: AI refinement for selected uncertain fuzzy matches only
    ai_calls_made = 0
    if args.use_ai_refinement == "true" and fast_matches and args.ai_api_key:
        print(f"Phase 2: AI refinement for top {len(fast_matches)} promising uncertain matches...")
        
        for col, fuzzy_field, match_type, score in fast_matches:
            refined_field, refined_type, refined_score = gemini_refine_match(
                col, fuzzy_field, available_fields, args.ai_api_key, args.ai_model
            )
            ai_calls_made += 1
            
            if refined_field and refined_field in available_fields:
                mapped_columns[col] = refined_field
                available_fields.remove(refined_field)
                
                tables = FIELD_TO_TABLE.get(refined_field, ["Unknown"])
                for table in tables:
                    table_summary[table].append({
                        "source_column": col,
                        "schema_field": refined_field,
                        "match_type": refined_type,
                        "score": refined_score
                    })
            else:
                # AI couldn't improve, use original fuzzy match if still available
                if fuzzy_field in available_fields:
                    mapped_columns[col] = fuzzy_field
                    available_fields.remove(fuzzy_field)
                    
                    tables = FIELD_TO_TABLE.get(fuzzy_field, ["Unknown"])
                    for table in tables:
                        table_summary[table].append({
                            "source_column": col,
                            "schema_field": fuzzy_field,
                            "match_type": "fuzzy_fallback",
                            "score": score
                        })
    else:
        # Skip AI, just use the fuzzy matches
        for col, field, match_type, score in fast_matches:
            if field in available_fields:
                mapped_columns[col] = field
                available_fields.remove(field)
                
                tables = FIELD_TO_TABLE.get(field, ["Unknown"])
                for table in tables:
                    table_summary[table].append({
                        "source_column": col,
                        "schema_field": field,
                        "match_type": match_type,
                        "score": score
                    })

    # Generate filtered DataFrame
    if mapped_columns:
        filtered_df = df[list(mapped_columns.keys())].copy()
        filtered_df.rename(columns=mapped_columns, inplace=True)
        filtered_df.to_csv(args.output, index=False)
    else:
        filtered_df = pd.DataFrame()
        print("No columns could be mapped!")

    # Display summary
    print("\n" + "="*60)
    print("HYBRID COLUMN FILTERING SUMMARY")
    print("="*60)
    print(f"Total input columns: {len(df.columns)}")
    print(f"Successfully mapped: {len(mapped_columns)}")
    print(f"Unmatched columns: {len(df.columns) - len(mapped_columns)}")
    print(f"AI calls made: {ai_calls_made} (only for uncertain fuzzy matches)")
    
    if mapped_columns:
        print(f"\nFiltered data shape: {filtered_df.shape}")
        print(f"Saved to: {os.path.abspath(args.output)}")
        
        print(f"\nMapped columns by schema table:")
        print("-" * 40)
        for table in sorted(table_summary.keys()):
            cols = table_summary[table]
            print(f"\n{table} ({len(cols)} columns):")
            for col_info in cols:
                print(f"  • {col_info['source_column']} → {col_info['schema_field']} "
                      f"({col_info['match_type']}, {col_info['score']:.3f})")
        
        # Show match type distribution
        match_types = defaultdict(int)
        for table_cols in table_summary.values():
            for col_info in table_cols:
                match_types[col_info['match_type']] += 1
        
        print(f"\nMatch Type Distribution:")
        for match_type, count in sorted(match_types.items()):
            print(f"  • {match_type}: {count} columns")
    
    print("\n" + "="*60)
    print("HYBRID FILTERING COMPLETE - Ready for next workflow step")
    print("="*60)

if __name__ == "__main__":
    main()
