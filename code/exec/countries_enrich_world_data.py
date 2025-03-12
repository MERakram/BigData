import pandas as pd
import os
import sys

# Add parent directory to Python path to find the utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_countries_json, load_csv_data
from utils.country_processor import (
    normalize_country_name,
    create_country_mappings,
    get_missed_countries_info,
)
from utils.output_handler import save_enriched_data, save_missing_report


def match_and_enrich(
    countries_data,
    world_data,
    name_to_index,
    code_to_index,
    capital_to_index,
    region_subregion_to_index,
):
    """Match countries between datasets and enrich the JSON data with CSV data"""
    # Track matches and misses
    matched_countries = set()
    missed_countries = {}
    enriched_count = 0

    # Identify the column name that contains country identifiers
    country_col = world_data.columns[0]

    # Process each row in world data
    for _, row in world_data.iterrows():
        country_matched = False

        country_name = row[country_col]
        if not pd.notna(country_name):
            continue

        original_name = country_name
        country_key = str(country_name).lower().strip()
        normalized_key = normalize_country_name(country_key)
        matched_index = None
        match_method = None

        # Try to match by name
        for name_key in [country_key, normalized_key]:
            if name_key in name_to_index:
                matched_index = name_to_index[name_key]
                match_method = "name_match"
                country_matched = True
                break

        # Try to match by code if matching by name failed
        if not country_matched:
            for code_key in [country_key, normalized_key]:
                if code_key in code_to_index:
                    matched_index = code_to_index[code_key]
                    match_method = "code_match"
                    country_matched = True
                    break

        # Try to match by capital if still no match
        if not country_matched:
            for cap_key in [country_key, normalized_key]:
                if cap_key in capital_to_index:
                    matched_index = capital_to_index[cap_key]
                    match_method = "capital_match"
                    country_matched = True
                    break

        # Try region-subregion as last resort
        if not country_matched and "region" in row and "subregion" in row:
            if pd.notna(row["region"]) and pd.notna(row["subregion"]):
                region_key = (
                    f"{str(row['region']).lower()}-{str(row['subregion']).lower()}"
                )
                if region_key in region_subregion_to_index:
                    matched_index = region_subregion_to_index[region_key]
                    match_method = "region_match"
                    country_matched = True

        # Add world data if matched
        if country_matched and matched_index is not None:
            if "world_data" not in countries_data[matched_index]:
                # Create a world data dictionary
                world_entry = {}
                for col in world_data.columns:
                    if col != country_col and pd.notna(row[col]):
                        world_entry[col] = row[col]

                # Add metadata
                world_entry["_match_method"] = match_method
                world_entry["_match_key"] = normalized_key
                world_entry["_original_world_name"] = original_name

                countries_data[matched_index]["world_data"] = world_entry
                matched_countries.add(original_name)
                enriched_count += 1
        else:
            # Record missed countries
            missed_countries[country_name] = "No match found"

    return countries_data


def main():
    # File paths
    countries_json_path = "../../data/input/countries.json"
    world_data_csv_path = "../../data/input/world-data-2023.csv"
    output_path = "../../data/output/enriched_countries.json"
    missing_report_path = "../../data/output/missing_countries_report.md"

    print("Starting country data enrichment process...")

    # Load data
    countries_data = load_countries_json(countries_json_path)
    if not countries_data:
        return

    world_data = load_csv_data(world_data_csv_path, "world")
    if world_data.empty:
        return

    # Create mappings for faster lookups
    name_to_index, code_to_index, capital_to_index, region_subregion_to_index = (
        create_country_mappings(countries_data)
    )

    # Match and enrich data
    enriched_data = match_and_enrich(
        countries_data,
        world_data,
        name_to_index,
        code_to_index,
        capital_to_index,
        region_subregion_to_index,
    )

    # Save the enriched data
    save_enriched_data(enriched_data, output_path)
    print(f"\n==================== Report ===================\n")

    # Generate missing countries report using the improved function
    missed_countries = get_missed_countries_info(enriched_data, "world_data")

    world_keys = set()
    for _, row in world_data.iterrows():
        country_col = world_data.columns[0]
        if pd.notna(row[country_col]):
            world_keys.add(str(row[country_col]).strip())

    save_missing_report(missed_countries, world_keys, missing_report_path, "world")

    print("Country data enrichment complete!")


if __name__ == "__main__":
    main()
