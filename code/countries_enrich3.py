import pandas as pd
import json
import os
import re
from collections import defaultdict
from fuzzywuzzy import fuzz, process


def load_countries_data(json_path):
    """Load countries data from the JSON file"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            countries_data = json.load(f)
        print(f"Successfully loaded {len(countries_data)} countries from JSON file")
        return countries_data
    except Exception as e:
        print(f"Error loading countries data: {str(e)}")
        return []


def load_world_data(csv_path):
    """Load world data from the CSV file"""
    try:
        world_data = pd.read_csv(csv_path)
        print(
            f"Successfully loaded world data with {len(world_data)} rows and {len(world_data.columns)} columns"
        )
        return world_data
    except Exception as e:
        print(f"Error loading world data: {str(e)}")
        return pd.DataFrame()


def normalize_country_name(name):
    """Normalize country names for better matching"""
    if not name:
        return ""

    name = name.lower().strip()

    # Remove common prefixes/suffixes
    prefixes = [
        "republic of ",
        "kingdom of ",
        "united republic of ",
        "democratic republic of ",
        "federation of ",
        "people's republic of ",
        "the ",
        "state of ",
        "sultanate of ",
        "principality of ",
        "commonwealth of ",
        "federal republic of ",
        "grand duchy of ",
        "union of ",
    ]

    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix) :]

    # Special handling for "South X" vs "X" - make sure we don't collapse them
    if name.startswith("south "):
        return name  # Don't normalize further to avoid confusion

    # Handle common name variations
    replacements = {
        "united states of america": "united states",
        "america": "united states",
        "usa": "united states",
        "u.s.a.": "united states",
        "u.s.": "united states",
        "uk": "united kingdom",
        "great britain": "united kingdom",
        "britain": "united kingdom",
        "myanmar": "burma",
        "democratic people's republic of korea": "north korea",
        "dprk": "north korea",
        "republic of korea": "south korea",
        "ivory coast": "côte d'ivoire",
        "eswatini": "swaziland",
        "czech republic": "czechia",
        "russian federation": "russia",
        "united arab emirates": "uae",
        "palestine": "palestinian territories",
        "viet nam": "vietnam",
        "korea, republic of": "south korea",
        "korea, democratic people's republic of": "north korea",
        "congo, democratic republic of the": "democratic republic of the congo",
        "congo, republic of the": "republic of congo",
        "congo-kinshasa": "democratic republic of the congo",
        "congo-brazzaville": "republic of congo",
        "tanzania, united republic of": "tanzania",
        "macedonia, the former yugoslav republic of": "north macedonia",
        "iran, islamic republic of": "iran",
        "micronesia, federated states of": "micronesia",
        "saint martin": "st. martin",
        "saint kitts and nevis": "st. kitts and nevis",
        "saint vincent and the grenadines": "st. vincent and the grenadines",
        "saint lucia": "st. lucia",
        "georgia": "georgia country",  # Explicitly differentiate
        "south georgia and the south sandwich islands": "south georgia",
        "south georgia": "south georgia island",  # Mark as island
        "guinea bissau": "guinea-bissau",
        "timor leste": "east timor",
        "vatican city state": "vatican city",
        "holy see": "vatican city",
        "hong kong, sar china": "hong kong",
        "macao, sar china": "macau",
        "virgin islands, british": "british virgin islands",
        "virgin islands, u.s.": "us virgin islands",
    }

    return replacements.get(name, name)


def extract_name_from_country(country):
    """Extract the name from a country object safely"""
    if "name" not in country:
        return ""

    if "common" in country["name"]:
        return country["name"]["common"]
    elif "official" in country["name"]:
        return country["name"]["official"]

    return ""


def get_all_country_names(country):
    """Extract all possible names for a country from its JSON data"""
    names = set()

    if "name" in country:
        if "common" in country["name"]:
            names.add(normalize_country_name(country["name"]["common"]))
            # Also add the non-normalized version
            names.add(country["name"]["common"].lower().strip())

        if "official" in country["name"]:
            names.add(normalize_country_name(country["name"]["official"]))
            names.add(country["name"]["official"].lower().strip())

        # Add native names
        if "nativeName" in country["name"]:
            for lang, name_data in country["name"]["nativeName"].items():
                if "common" in name_data:
                    names.add(normalize_country_name(name_data["common"]))
                if "official" in name_data:
                    names.add(normalize_country_name(name_data["official"]))

    # Add alternative spellings
    if "altSpellings" in country:
        for alt in country["altSpellings"]:
            names.add(normalize_country_name(alt))
            # Special cases for Georgia
            if "georgia" in alt.lower() and "south" not in alt.lower():
                names.add("georgia country")

    # Add translations if they exist
    if "translations" in country:
        for lang, translation in country["translations"].items():
            if isinstance(translation, dict) and "common" in translation:
                names.add(normalize_country_name(translation["common"]))

    # Add special cases for commonly confused countries
    country_name = extract_name_from_country(country)
    if "Georgia" == country_name and "South" not in country_name:
        names.add("georgia country")
        names.add("georgia nation")
    elif "South Georgia" in country_name:
        names.add("south georgia island")
        names.add("south georgia and sandwich")

    # Remove empty strings
    names.discard("")
    return names


def create_country_mappings(countries_data):
    """Create mappings from various country identifiers to their index in the countries_data array"""
    # Initialize mapping dictionaries
    name_to_index = {}
    code_to_index = {}
    capital_to_index = {}
    region_subregion_to_index = {}

    for i, country in enumerate(countries_data):
        # Get all possible names for this country
        all_names = get_all_country_names(country)
        for name in all_names:
            if name:  # Skip empty names
                name_to_index[name] = i

        # Map country codes to index
        for code_field in ["cca2", "cca3", "ccn3"]:
            if code_field in country and country[code_field]:
                code = str(country[code_field]).lower()
                code_to_index[code] = i

        # Map capital cities to index
        if "capital" in country and country["capital"]:
            for capital in country["capital"]:
                capital_norm = capital.lower().strip()
                capital_to_index[capital_norm] = i

        # Map region + subregion to help with disambiguation
        if "region" in country and "subregion" in country:
            region = country["region"].lower() if country["region"] else ""
            subregion = country["subregion"].lower() if country["subregion"] else ""
            if region and subregion:
                key = f"{region}:{subregion}"
                if key not in region_subregion_to_index:
                    region_subregion_to_index[key] = []
                region_subregion_to_index[key].append(i)

    return name_to_index, code_to_index, capital_to_index, region_subregion_to_index


def create_manual_mapping():
    """Create a manual mapping for commonly mismatched countries"""
    return {
        # United States variations
        "us": "united states",
        "u.s.": "united states",
        "united states": "united states",
        "united states of america": "united states",
        "america": "united states",
        "usa": "united states",
        "u.s.a.": "united states",
        # United Kingdom variations
        "uk": "united kingdom",
        "britain": "united kingdom",
        "great britain": "united kingdom",
        "england": "united kingdom",  # Not exactly correct but common usage
        # Georgia vs South Georgia
        "georgia": "georgia country",
        "georgia (country)": "georgia country",
        "georgia country": "georgia country",
        "georgia nation": "georgia country",
        "south georgia": "south georgia island",
        "south georgia island": "south georgia island",
        "south georgia and the south sandwich islands": "south georgia island",
        # UAE
        "uae": "united arab emirates",
        "u.a.e": "united arab emirates",
        "emirates": "united arab emirates",
        # Russia
        "russia": "russian federation",
        "russian federation": "russian federation",
        # Special territories
        "taiwan": "taiwan, province of china",
        "taiwan, china": "taiwan, province of china",
        "chinese taipei": "taiwan, province of china",
        # China
        "china": "people's republic of china",
        "prc": "people's republic of china",
        "mainland china": "people's republic of china",
        # Korea
        "north korea": "democratic people's republic of korea",
        "nk": "democratic people's republic of korea",
        "south korea": "republic of korea",
        "korea": "republic of korea",  # Default to South Korea if ambiguous
        # Congo
        "drc": "democratic republic of the congo",
        "congo, the democratic republic of the": "democratic republic of the congo",
        "dr congo": "democratic republic of the congo",
        "congo-kinshasa": "democratic republic of the congo",
        "republic of congo": "congo",
        "congo, republic of": "congo",
        "congo-brazzaville": "congo",
        "congo republic": "congo",
        # Common name changes
        "ivory coast": "côte d'ivoire",
        "swaziland": "eswatini",
        "czechia": "czech republic",
        "burma": "myanmar",
        "macedonia": "north macedonia",
        # Other confusions
        "bosnia": "bosnia and herzegovina",
        "vietnam": "viet nam",
        "laos": "lao people's democratic republic",
        "venezuela": "venezuela, bolivarian republic of",
        "bolivia": "bolivia, plurinational state of",
        "syria": "syrian arab republic",
        "palestine": "palestinian territories",
        "palestine, state of": "palestinian territories",
        "east timor": "timor-leste",
        "brunei": "brunei darussalam",
        # Islands and territories
        "virgin islands": "british virgin islands",
        "virgin islands, british": "british virgin islands",
        "virgin islands, us": "united states virgin islands",
        "virgin islands, u.s.": "united states virgin islands",
        "us virgin islands": "united states virgin islands",
        "cayman islands": "cayman islands",
        "falkland islands": "falkland islands (malvinas)",
        "falklands": "falkland islands (malvinas)",
        # Common spelling variations
        "guinea bissau": "guinea-bissau",
        "st. kitts and nevis": "saint kitts and nevis",
        "st kitts and nevis": "saint kitts and nevis",
        "st. vincent": "saint vincent and the grenadines",
        "st vincent": "saint vincent and the grenadines",
    }


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
    missed_countries = {}  # Will store country name and potential reasons
    enriched_count = 0

    # Get the column that contains country names/codes in world_data
    country_col = None
    for col in world_data.columns:
        if col.lower() in ["country", "country_name", "name", "nation", "country_code"]:
            country_col = col
            break

    if not country_col:
        country_col = world_data.columns[
            0
        ]  # Default to first column if no obvious name column

    print(f"Using '{country_col}' as the country identifier column")

    # Create manual mapping for problematic countries
    manual_mapping = create_manual_mapping()

    # Create a dictionary from world_data for faster lookups
    world_dict = defaultdict(dict)
    world_keys = set()
    world_keys_original = {}  # Store original names for debugging

    for _, row in world_data.iterrows():
        country = row[country_col]
        if pd.notna(country):
            original_key = str(country).strip()
            country_key = original_key.lower()
            normalized_key = normalize_country_name(country_key)

            # Store data under both original and normalized keys
            for key in [country_key, normalized_key]:
                if key:  # Skip empty keys
                    world_keys.add(key)
                    world_keys_original[key] = original_key
                    for col in world_data.columns:
                        if col != country_col and pd.notna(row[col]):
                            world_dict[key][col] = row[col]

    # Store world data keys for debugging
    print(f"Found {len(world_keys)} unique country identifiers in the world data")

    # Enrich each country with world data
    for i, country in enumerate(countries_data):
        country_matched = False
        country_name = extract_name_from_country(country)
        matched_key = None
        match_method = None

        # Try to match by name
        if "name" in country:
            country_names = get_all_country_names(country)

            # Try direct matches first
            for name in country_names:
                if name in world_dict:
                    matched_key = name
                    country_matched = True
                    match_method = "direct_name_match"
                    break

                # Try manual mapping
                mapped_name = manual_mapping.get(name)
                if mapped_name and mapped_name in world_dict:
                    matched_key = mapped_name
                    country_matched = True
                    match_method = "manual_mapping"
                    break

            # Try code matching if name matching failed
            if not country_matched:
                for code_field in ["cca2", "cca3", "ccn3"]:
                    if code_field in country and country[code_field]:
                        code = str(country[code_field]).lower()
                        if code in world_dict:
                            matched_key = code
                            country_matched = True
                            match_method = f"code_match_{code_field}"
                            break

            # Try capital city matching
            if not country_matched and "capital" in country and country["capital"]:
                capital = country["capital"][0].lower()
                # Look for countries with this capital
                for world_key in world_keys:
                    if capital in world_key or world_key in capital:
                        matched_key = world_key
                        country_matched = True
                        match_method = "capital_match"
                        break

            # If no direct match, try fuzzy matching as a last resort
            if not country_matched and country_names:
                primary_name = next(iter(country_names))
                # Only consider the top match with a high score
                best_match = process.extractOne(
                    primary_name,
                    world_keys,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=85,
                )
                if best_match:
                    matched_key = best_match[0]
                    country_matched = True
                    match_method = f"fuzzy_match_{best_match[1]}"

        # Add world data if matched
        if country_matched and matched_key:
            countries_data[i]["world_data"] = dict(world_dict[matched_key])
            # Add metadata about the match
            countries_data[i]["world_data"]["_match_method"] = match_method
            countries_data[i]["world_data"]["_match_key"] = matched_key
            countries_data[i]["world_data"]["_original_world_name"] = (
                world_keys_original.get(matched_key, matched_key)
            )

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

    print(f"Enriched {enriched_count} countries with world data")
    print(f"Missed {len(missed_countries)} countries")

    # Print some missed countries for debugging
    if missed_countries:
        print("\nMissed countries (showing up to 10):")
        for i, (country, reason) in enumerate(list(missed_countries.items())[:10]):
            print(f"  - {country}: {reason}")

    return countries_data


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


def save_missing_report(missed_countries, world_keys, output_path):
    """Save a report of missing countries and available world data keys"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Missing Countries Report\n\n")
            f.write("## Countries that could not be matched:\n\n")
            for country, reason in missed_countries.items():
                f.write(f"- {country}: {reason}\n")

            f.write("\n\n## Available keys in world data:\n\n")
            for key in sorted(world_keys):
                f.write(f"- {key}\n")

        print(f"Saved missing countries report to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving missing report: {str(e)}")
        return False


def main():
    # File paths
    countries_json_path = "../data/countries/countries.json"
    world_data_csv_path = "../data/countries/world-data-2023.csv"
    output_path = "../data/output/enriched_countries.json"
    missing_report_path = "../data/output/missing_countries_report.md"

    print("Starting country data enrichment process...")

    # Load data
    countries_data = load_countries_data(countries_json_path)
    if not countries_data:
        return

    world_data = load_world_data(world_data_csv_path)
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

    # Generate missing countries report
    missed_countries = {}
    for country in enriched_data:
        if (
            "world_data" not in country
            and "name" in country
            and "common" in country["name"]
        ):
            name = country["name"]["common"]
            missed_countries[name] = "No match found"

    world_keys = set()
    for _, row in world_data.iterrows():
        country_col = world_data.columns[0]
        if pd.notna(row[country_col]):
            world_keys.add(str(row[country_col]).strip())

    save_missing_report(missed_countries, world_keys, missing_report_path)

    print("Country data enrichment complete!")


if __name__ == "__main__":
    main()
