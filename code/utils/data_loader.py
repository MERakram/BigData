import pandas as pd
import json


def load_countries_json(json_path):
    """Load countries data from JSON file"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            countries_data = json.load(f)
        print(f"Successfully loaded {len(countries_data)} countries from JSON file")
        return countries_data
    except Exception as e:
        print(f"Error loading countries data: {str(e)}")
        return []


def load_csv_data(csv_path, data_type="CSV"):
    """Load data from CSV file with custom data type label for messages"""
    try:
        data = pd.read_csv(csv_path)
        print(
            f"Successfully loaded {data_type} data with {len(data)} rows and {len(data.columns)} columns"
        )
        return data
    except Exception as e:
        print(f"Error loading {data_type} data: {str(e)}")
        return pd.DataFrame()
