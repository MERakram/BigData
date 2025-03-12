import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_countries_json, load_csv_data
from utils.country_processor import (
    normalize_country_name,
    extract_name_from_country,
    match_country_with_dataset,
    get_missed_countries_info,
)
from utils.output_handler import save_enriched_data, save_missing_report


def process_unesco_data(unesco_data):
    """Process UNESCO data into a dictionary for lookups"""
    # Identify the column name that contains country identifiers
    country_col = None
    for col in unesco_data.columns:
        if col.lower() in [
            "country",
            "country_name",
            "name",
            "nation",
            "country or region",
        ]:
            country_col = col
            break

    if not country_col:
        country_col = unesco_data.columns[
            0
        ]  # Default to first column if no obvious name column

    print(f"Using '{country_col}' as the country identifier column in UNESCO data")

    # Convert UNESCO data to dict for faster lookup
    unesco_dict = {}
    unesco_keys = set()

    for _, row in unesco_data.iterrows():
        country = row[country_col]
        if pd.notna(country):
            country_key = str(country).lower().strip()
            normalized_key = normalize_country_name(country_key)

            # Create unesco data dictionary
            unesco_entry = {}
            for col in unesco_data.columns:
                if col != country_col and pd.notna(row[col]):
                    unesco_entry[col] = row[col]

            # Add entry under both original and normalized keys
            for key in [country_key, normalized_key]:
                if key:  # Skip empty keys
                    unesco_dict[key] = unesco_entry
                    unesco_keys.add(key)

    return unesco_dict, unesco_keys, country_col


def match_and_enrich_unesco(countries_data, unesco_data):
    """Match countries between datasets and enrich with UNESCO data"""
    # Track matches and misses
    matched_countries = set()
    missed_countries = {}
    enriched_count = 0

    # Process UNESCO data
    unesco_dict, unesco_keys, country_col = process_unesco_data(unesco_data)
    print(f"Found {len(unesco_keys)} unique country identifiers in UNESCO data")

    # Enrich each country with UNESCO data
    for i, country in enumerate(countries_data):
        country_name = extract_name_from_country(country)
        matched_key, match_method = match_country_with_dataset(
            country, unesco_dict, unesco_keys
        )

        # Add UNESCO data if matched
        if matched_key:
            if "unesco_data" not in countries_data[i]:
                countries_data[i]["unesco_data"] = dict(unesco_dict[matched_key])
                # Add metadata about the match
                countries_data[i]["unesco_data"]["_match_method"] = match_method
                countries_data[i]["unesco_data"]["_match_key"] = matched_key

                matched_countries.add(matched_key)
                enriched_count += 1
        else:
            # Record misses
            if country_name:
                missed_countries[country_name] = "No match found"

    print(f"Enriched {enriched_count} countries with UNESCO data")
    print(f"Missed {len(missed_countries)} countries")

    # Print some missed countries for debugging
    if missed_countries:
        print("\nMissed countries (showing up to 10):")
        for i, (country, reason) in enumerate(list(missed_countries.items())[:10]):
            print(f"  - {country}: {reason}")

    return countries_data, country_col


def main():
    # File paths
    enriched_json_path = "../../data/output/enriched_countries_happiness.json"
    unesco_csv_path = "../../data/input/unesco-heritage-sites-2019.csv"
    output_path = "../../data/output/enriched_countries_final.json"
    missing_report_path = "../../data/output/missing_unesco_report.md"

    print("Starting country data enrichment with UNESCO data...")

    # Load data
    countries_data = load_countries_json(enriched_json_path)
    if not countries_data:
        return

    unesco_data = load_csv_data(unesco_csv_path, "UNESCO")
    if unesco_data.empty:
        return

    # Match and enrich data
    enriched_data, country_col = match_and_enrich_unesco(countries_data, unesco_data)

    # Save the enriched data
    save_enriched_data(enriched_data, output_path)

    # Generate missing UNESCO data report
    missed_countries = get_missed_countries_info(enriched_data, "unesco_data")

    unesco_keys = set()
    for _, row in unesco_data.iterrows():
        if pd.notna(row[country_col]):
            unesco_keys.add(str(row[country_col]).strip())

    save_missing_report(missed_countries, unesco_keys, missing_report_path, "UNESCO")

    print("Country data enrichment with UNESCO data complete!")


if __name__ == "__main__":
    main()
