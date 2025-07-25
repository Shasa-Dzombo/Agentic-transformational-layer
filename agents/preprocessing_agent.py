import pandas as pd
from tools.data_dictionary import DataDictionary
from schema.introspect import SchemaInspector

class PreprocessingAgent:
    def __init__(self, data_dict_path: str, db_engine):
        self.data_dict = DataDictionary(data_dict_path)
        self.schema_inspector = SchemaInspector(db_engine)

    def preprocess_prompt(self, prompt: str) -> dict:
        variables = self.extract_variables(prompt)

        validated = {
            var: self.data_dict.lookup(var) 
            for var in variables 
            if self.data_dict.lookup(var) is not None
        }

        table_info = {
            var: self.schema_inspector.find_column_location(var)
            for var in validated
        }

        checks = {
            var: self.basic_checks(table_info[var], var)
            for var in validated if table_info.get(var)
        }

        return {
            "original_prompt": prompt,
            "validated_variables": validated,
            "schema_context": table_info,
            "sanity_checks": checks
        }

    def extract_variables(self, prompt: str) -> list:
        words = prompt.replace(".", "").split()
        return [word for word in words if word.lower() in self.data_dict.dictionary]

    def basic_checks(self, table: str, column: str) -> dict:
        query = f"SELECT `{column}` FROM `{table}` LIMIT 1000"
        df = pd.read_sql(query, self.schema_inspector.engine)

        return {
            "null_rate": round(df[column].isnull().mean(), 4),
            "unique_count": int(df[column].nunique()),
            "dtype": str(df[column].dtype)
        }

