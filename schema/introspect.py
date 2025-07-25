from sqlalchemy import inspect

class SchemaInspector:
    def __init__(self, engine):
        self.engine = engine
        self.inspector = inspect(engine)

    def find_column_location(self, column_name):
        for table_name in self.inspector.get_table_names():
            columns = [col["name"] for col in self.inspector.get_columns(table_name)]
            if column_name in columns:
                return table_name
        return None