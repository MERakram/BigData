import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_countries_json, load_csv_data
from utils.country_processor import (
    normalize_country_name,
    extract_name_from_country,
    get_all_country_names,
    match_country_with_dataset,
    get_missed_countries_info,
)
from utils.output_handler import save_enriched_data, save_missing_report


def process_happiness_data(happiness_data):
    """Process happiness data into a dictionary for lookups"""
    happiness_dict = {}
    happiness_keys = set()

    for _, row in happiness_data.iterrows():
        country = row["Country or region"]
        if pd.notna(country):
            country_key = country.lower().strip()
            normalized_key = normalize_country_name(country_key)

            # Create happiness data dictionary
            happiness_entry = {}
            for col in happiness_data.columns:
                if col != "Country or region" and pd.notna(row[col]):
                    happiness_entry[col] = row[col]

            # Add entry under both original and normalized keys
            for key in [country_key, normalized_key]:
                if key:  # Skip empty keys
                    happiness_dict[key] = happiness_entry
                    happiness_keys.add(key)

    return happiness_dict, happiness_keys


def match_and_enrich_happiness(countries_data, happiness_data):
    """Match countries between datasets and enrich with happiness data"""
    # Track matches and misses
    matched_countries = set()
    missed_countries = {}
    enriched_count = 0

    # Process happiness data
    happiness_dict, happiness_keys = process_happiness_data(happiness_data)
    print(f"Found {len(happiness_keys)} unique country identifiers in happiness data")

    # Enrich each country with happiness data
    for i, country in enumerate(countries_data):
        country_name = extract_name_from_country(country)
        matched_key, match_method = match_country_with_dataset(
            country, happiness_dict, happiness_keys
        )

        # Add happiness data if matched
        if matched_key:
            if "happiness_data" not in countries_data[i]:
                countries_data[i]["happiness_data"] = dict(happiness_dict[matched_key])
                # Add metadata about the match
                countries_data[i]["happiness_data"]["_match_method"] = match_method
                countries_data[i]["happiness_data"]["_match_key"] = matched_key

                matched_countries.add(matched_key)
                enriched_count += 1
        else:
            # Record misses with potential reasons
            if country_name:
                reason = "No match found"
                if "name" in country:
                    names = list(get_all_country_names(country))
                    if len(names) > 0:
                        reason = f"Tried names: {', '.join(names[:3])}"
                        if len(names) > 3:
                            reason += f" (and {len(names) - 3} more)"
                missed_countries[country_name] = reason

    print(f"Enriched {enriched_count} countries with happiness data")
    print(f"Missed {len(missed_countries)} countries")

    # Print some missed countries for debugging
    if missed_countries:
        print("\nMissed countries (showing up to 10):")
        for i, (country, reason) in enumerate(list(missed_countries.items())[:10]):
            # print(f"  - {country}: {reason}")
            print(f"  - {country}")

    return countries_data


def main():
    # File paths
    enriched_json_path = "../../data/output/enriched_countries.json"
    happiness_csv_path = "../../data/input/hapiness_2019.csv"
    output_path = "../../data/output/enriched_countries_happiness.json"
    missing_report_path = "../../data/output/missing_happiness_report.md"

    print("Starting country data enrichment with happiness data...")

    # Load data
    countries_data = load_countries_json(enriched_json_path)
    if not countries_data:
        return

    happiness_data = load_csv_data(happiness_csv_path, "happiness")
    if happiness_data.empty:
        return

    # Match and enrich data
    enriched_data = match_and_enrich_happiness(countries_data, happiness_data)

    # Save the enriched data
    save_enriched_data(enriched_data, output_path)
    print(f"\n==================== Report ===================\n")

    # Generate missing happiness data report
    missed_countries = get_missed_countries_info(enriched_data, "happiness_data")

    happiness_keys = set()
    for _, row in happiness_data.iterrows():
        if pd.notna(row["Country or region"]):
            happiness_keys.add(str(row["Country or region"]).strip())

    save_missing_report(
        missed_countries, happiness_keys, missing_report_path, "happiness"
    )

    print("Country data enrichment with happiness data complete!")


if __name__ == "__main__":
    main()
