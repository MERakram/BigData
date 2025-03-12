import sys
import os
import json

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_countries_json
from utils.output_handler import save_enriched_data


def merge_enriched_data(
    base_data, world_data=None, happiness_data=None, unesco_data=None
):
    """Merge multiple enriched datasets into a single comprehensive dataset"""

    # Initialize result with base data
    result = base_data.copy()

    # Track enrichment counts
    enriched_count = {"world": 0, "happiness": 0, "unesco": 0}

    # Create index mappings for faster lookups
    base_name_to_index = {}
    base_cca2_to_index = {}
    base_cca3_to_index = {}

    for i, country in enumerate(result):
        if "name" in country and "common" in country["name"]:
            base_name_to_index[country["name"]["common"].lower()] = i
        if "cca2" in country:
            base_cca2_to_index[country["cca2"].lower()] = i
        if "cca3" in country:
            base_cca3_to_index[country["cca3"].lower()] = i

    # Merge world data if provided
    if world_data:
        for country in world_data:
            matched_index = None

            # Try to match by name
            if "name" in country and "common" in country["name"]:
                name_key = country["name"]["common"].lower()
                if name_key in base_name_to_index:
                    matched_index = base_name_to_index[name_key]

            # Try to match by code
            if matched_index is None and "cca2" in country:
                if country["cca2"].lower() in base_cca2_to_index:
                    matched_index = base_cca2_to_index[country["cca2"].lower()]

            if matched_index is None and "cca3" in country:
                if country["cca3"].lower() in base_cca3_to_index:
                    matched_index = base_cca3_to_index[country["cca3"].lower()]

            # If matched, copy world_data
            if matched_index is not None and "world_data" in country:
                result[matched_index]["world_data"] = country["world_data"]
                enriched_count["world"] += 1

    # Merge happiness data if provided
    if happiness_data:
        for country in happiness_data:
            matched_index = None

            # Try to match by name
            if "name" in country and "common" in country["name"]:
                name_key = country["name"]["common"].lower()
                if name_key in base_name_to_index:
                    matched_index = base_name_to_index[name_key]

            # Try to match by code
            if matched_index is None and "cca2" in country:
                if country["cca2"].lower() in base_cca2_to_index:
                    matched_index = base_cca2_to_index[country["cca2"].lower()]

            if matched_index is None and "cca3" in country:
                if country["cca3"].lower() in base_cca3_to_index:
                    matched_index = base_cca3_to_index[country["cca3"].lower()]

            # If matched, copy happiness_data
            if matched_index is not None and "happiness_data" in country:
                result[matched_index]["happiness_data"] = country["happiness_data"]
                enriched_count["happiness"] += 1

    # Merge UNESCO data if provided
    if unesco_data:
        for country in unesco_data:
            matched_index = None

            # Try to match by name
            if "name" in country and "common" in country["name"]:
                name_key = country["name"]["common"].lower()
                if name_key in base_name_to_index:
                    matched_index = base_name_to_index[name_key]

            # Try to match by code
            if matched_index is None and "cca2" in country:
                if country["cca2"].lower() in base_cca2_to_index:
                    matched_index = base_cca2_to_index[country["cca2"].lower()]

            if matched_index is None and "cca3" in country:
                if country["cca3"].lower() in base_cca3_to_index:
                    matched_index = base_cca3_to_index[country["cca3"].lower()]

            # If matched, copy unesco_sites and unesco_meta
            if matched_index is not None:
                if "unesco_sites" in country:
                    result[matched_index]["unesco_sites"] = country["unesco_sites"]
                if "unesco_meta" in country:
                    result[matched_index]["unesco_meta"] = country["unesco_meta"]
                    enriched_count["unesco"] += 1

    print(f"Enriched {enriched_count['world']} countries with world data")
    print(f"Enriched {enriched_count['happiness']} countries with happiness data")
    print(f"Enriched {enriched_count['unesco']} countries with UNESCO data")

    return result


def main():
    """Main function to merge all enriched data"""
    base_path = "../../data/input/countries.json"
    world_path = "../../data/output/enriched_countries.json"
    happiness_path = "../../data/output/enriched_countries_happiness.json"
    unesco_path = "../../data/output/enriched_countries_unesco.json"
    final_output_path = "../../data/output/enriched_countries_final.json"

    print("Starting merging of all enriched country data...")

    # Load data
    base_data = load_countries_json(base_path)
    world_data = load_countries_json(world_path)
    happiness_data = load_countries_json(happiness_path)
    unesco_data = load_countries_json(unesco_path)

    # Merge data
    merged_data = merge_enriched_data(
        base_data, world_data, happiness_data, unesco_data
    )

    # Save merged data
    save_enriched_data(merged_data, final_output_path)

    print(f"Successfully merged all enriched data into {final_output_path}")


if __name__ == "__main__":
    main()
