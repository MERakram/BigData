import os
import json


def save_enriched_data(enriched_data, output_path):
    """Save enriched data to a new JSON file"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched_data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved enriched data to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving enriched data: {str(e)}")
        return False


def save_missing_report(missed_countries, data_keys, output_path, data_type="data"):
    """Save a report of missing countries and available data keys"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Missing {data_type.capitalize()} Data Report\n\n")
            f.write(
                f"## Countries that could not be matched with {data_type} data:\n\n"
            )
            for country, reason in missed_countries.items():
                f.write(f"- {country}: {reason}\n")

            f.write(f"\n\n## Available keys in {data_type} data:\n\n")
            for key in sorted(data_keys):
                f.write(f"- {key}\n")

        print(f"Saved missing {data_type} data report to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving missing report: {str(e)}")
        return False
