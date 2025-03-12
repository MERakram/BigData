import sys
import os
import pandas as pd
import json

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_countries_json, load_csv_data
from utils.country_processor import normalize_country_name, get_missed_countries_info
from utils.output_handler import save_enriched_data, save_missing_report


def create_country_mappings(countries_data):
    """Create mapping from country names to indices"""
    country_to_index = {}

    for i, country in enumerate(countries_data):
        if "name" in country and "common" in country["name"]:
            # Add both original and normalized country name for better matching
            name = country["name"]["common"]
            country_to_index[name.lower()] = i

            # Also add normalized version
            normalized_name = normalize_country_name(name.lower())
            if normalized_name:
                country_to_index[normalized_name] = i

            # Add official name as well if available
            if "official" in country["name"]:
                official = country["name"]["official"]
                country_to_index[official.lower()] = i
                normalized_official = normalize_country_name(official.lower())
                if normalized_official:
                    country_to_index[normalized_official] = i

    return country_to_index


def match_and_enrich(countries_data, unesco_data, country_to_index):
    """Match countries with UNESCO heritage sites and enrich data"""
    matched_countries = set()
    missed_countries = {}
    enriched_count = 0

    # Extract the country column (states_name_en) from UNESCO data
    country_col = unesco_data.columns[1]  # Second column (index 1) is states_name_en

    # Group UNESCO sites by country
    grouped_sites = {}
    for _, row in unesco_data.iterrows():
        country_name = row[country_col] if pd.notna(row[country_col]) else ""
        if not country_name:
            continue

        # Handle transboundary sites (multiple countries separated by commas)
        if "," in country_name:
            country_list = [c.strip() for c in country_name.split(",")]
            transboundary = True
        else:
            country_list = [country_name]
            transboundary = False

        # Create a dictionary with relevant UNESCO site data
        site_data = {
            "name": row["name_en"] if pd.notna(row["name_en"]) else "",
            "category": row["category"] if pd.notna(row["category"]) else "",
            "description": (
                row["short_description_en"]
                if pd.notna(row["short_description_en"])
                else ""
            ),
            "date_inscribed": (
                row["date_inscribed"] if pd.notna(row["date_inscribed"]) else ""
            ),
            "latitude": row["latitude"] if pd.notna(row["latitude"]) else None,
            "longitude": row["longitude"] if pd.notna(row["longitude"]) else None,
            "area_hectares": (
                row["area_hectares"] if pd.notna(row["area_hectares"]) else None
            ),
            "criteria": row["criteria_txt"] if pd.notna(row["criteria_txt"]) else "",
            "transboundary": transboundary,
            "shared_with": country_list if transboundary else [],
        }

        # Add site to each country in the list
        for single_country in country_list:
            if single_country not in grouped_sites:
                grouped_sites[single_country] = []
            grouped_sites[single_country].append(site_data)

    # Match UNESCO sites with countries
    for country_name, sites in grouped_sites.items():
        country_matched = False
        matched_index = None

        # Try to match by country name
        country_key = country_name.lower().strip()
        normalized_key = normalize_country_name(country_key)

        # Try both original and normalized names
        for name_key in [country_key, normalized_key]:
            if name_key in country_to_index:
                matched_index = country_to_index[name_key]
                country_matched = True
                break

        # Add UNESCO data if matched
        if country_matched and matched_index is not None:
            if "unesco_sites" not in countries_data[matched_index]:
                countries_data[matched_index]["unesco_sites"] = sites
                countries_data[matched_index]["unesco_meta"] = {
                    "_match_method": "country_name_match",
                    "_match_key": normalized_key,
                    "_original_unesco_country": country_name,
                    "_site_count": len(sites),
                }
                matched_countries.add(country_name)
                enriched_count += 1
            else:
                # If country already has unesco_sites, append to it
                countries_data[matched_index]["unesco_sites"].extend(sites)
                countries_data[matched_index]["unesco_meta"]["_site_count"] += len(
                    sites
                )

        else:
            # Record missed countries
            missed_countries[country_name] = "No match found"

    print(f"Enriched {enriched_count} countries with UNESCO heritage sites data")
    print(f"Missed {len(missed_countries)} UNESCO countries")

    # Print some missed countries for debugging
    if missed_countries:
        print("\nMissed UNESCO countries (showing up to 10):")
        for i, (country, reason) in enumerate(list(missed_countries.items())[:10]):
            print(f"  - {country}: {reason}")

    return countries_data


def main():
    # File paths
    countries_json_path = "../../data/input/countries.json"
    unesco_csv_path = "../../data/input/unesco-heritage-sites-2019.csv"
    output_path = "../../data/output/enriched_countries_unesco.json"
    missing_report_path = "../../data/output/missing_unesco_report.md"

    print("Starting country data enrichment with UNESCO heritage sites...")

    # Load data
    countries_data = load_countries_json(countries_json_path)
    if not countries_data:
        return

    unesco_data = load_csv_data(unesco_csv_path, "UNESCO heritage sites")
    if unesco_data.empty:
        return

    # Create mappings for faster lookups
    country_to_index = create_country_mappings(countries_data)
    print(f"Created lookup map with {len(country_to_index)} country name mappings")

    # Match and enrich data
    enriched_data = match_and_enrich(countries_data, unesco_data, country_to_index)

    # Save the enriched data
    save_enriched_data(enriched_data, output_path)
    print(f"\n==================== Report ===================\n")

    # Generate missing countries report
    missed_countries = get_missed_countries_info(enriched_data, "unesco_sites")

    unesco_countries = set()
    country_col = unesco_data.columns[1]  # states_name_en column
    for _, row in unesco_data.iterrows():
        if pd.notna(row[country_col]):
            unesco_countries.add(str(row[country_col]).strip())

    save_missing_report(
        missed_countries, unesco_countries, missing_report_path, "UNESCO"
    )

    print("Country data enrichment with UNESCO heritage sites complete!")


if __name__ == "__main__":
    main()
