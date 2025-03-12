from fuzzywuzzy import fuzz, process


def normalize_country_name(name):
    """Normalize country names for better matching"""
    if not name:
        return ""

    name = name.lower().strip()

    # Comprehensive list of replacements for better matching
    replacements = {
        # USA variants
        "united states of america": "united states",
        "america": "united states",
        "usa": "united states",
        "u.s.a.": "united states",
        "u.s.": "united states",
        # UK variants
        "uk": "united kingdom",
        "great britain": "united kingdom",
        "britain": "united kingdom",
        # Common name variants
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
        # Fixes for missing countries
        "the bahamas": "bahamas",
        "bahamas": "bahamas",
        "the gambia": "gambia",
        "gambia": "gambia",
        "palestinian national authority": "palestine",
        "palestinian territories": "palestine",
        "palestine": "palestine",
        # Additional mappings for UNESCO countries
        "bolivia (plurinational state of)": "bolivia",
        "plurinational state of bolivia": "bolivia",
        "estado plurinacional de bolivia": "bolivia",
        "cabo verde": "cape verde",
        "cape verde": "cabo verde",
        "gambia (the)": "gambia",
        "the gambia": "gambia",
        "iran (islamic republic of)": "iran",
        "islamic republic of iran": "iran",
        "micronesia (federated states of)": "micronesia",
        "federated states of micronesia": "micronesia",
        "venezuela (bolivarian republic of)": "venezuela",
        "bolivarian republic of venezuela": "venezuela",
        "república bolivariana de venezuela": "venezuela",
        # Special case
        "jerusalem (site proposed by jordan)": "jerusalem",
        "jerusalem": "israel",  # Consider the political implications of this mapping
        # Fix other problematic country names
        "north macedonia": "macedonia",
        "macedonia": "north macedonia",
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
            names.add(normalize_country_name(country["name"]["common"].lower().strip()))
            names.add(country["name"]["common"].lower().strip())

        if "official" in country["name"]:
            names.add(
                normalize_country_name(country["name"]["official"].lower().strip())
            )
            names.add(country["name"]["official"].lower().strip())

        # Add native names
        if "nativeName" in country["name"]:
            for lang, name_data in country["name"]["nativeName"].items():
                if "common" in name_data:
                    names.add(
                        normalize_country_name(name_data["common"].lower().strip())
                    )
                if "official" in name_data:
                    names.add(
                        normalize_country_name(name_data["official"].lower().strip())
                    )

    # Add alternative spellings
    if "altSpellings" in country:
        for alt in country["altSpellings"]:
            names.add(normalize_country_name(alt.lower().strip()))

    # Remove empty strings
    names.discard("")
    return names


def match_country_with_dataset(country, data_dict, data_keys):
    """Generic function to match a country with a dataset using different methods"""
    if "name" not in country:
        return None, None

    matched_key = None
    match_method = None
    country_names = get_all_country_names(country)

    # Try direct matches first
    for name in country_names:
        if name in data_dict:
            matched_key = name
            match_method = "direct_name_match"
            return matched_key, match_method

    # If no direct match, try fuzzy matching as a last resort
    if country_names:
        primary_name = next(iter(country_names))
        # Only consider the top match with a high score
        best_match = process.extractOne(
            primary_name,
            data_keys,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=85,
        )
        if best_match:
            matched_key = best_match[0]
            match_method = f"fuzzy_match_{best_match[1]}"
            return matched_key, match_method

    return None, None


def get_missed_countries_info(countries_data, data_field):
    """Get information about countries that couldn't be matched"""
    missed_countries = {}
    total_countries = len(countries_data)
    matched_count = sum(1 for country in countries_data if data_field in country)
    missing_count = total_countries - matched_count

    print(f"Total countries in dataset: {total_countries}")
    print(f"Successfully matched: {matched_count}")
    print(f"Missing matches: {missing_count}")

    # Collect individual missing country names for the report
    for country in countries_data:
        country_name = None
        if data_field not in country:
            if "name" in country:
                if "common" in country["name"]:
                    country_name = country["name"]["common"]
                elif "official" in country["name"]:
                    country_name = country["name"]["official"]

            # Handle countries with no proper name structure
            if not country_name and "cca2" in country:
                country_name = f"Country code: {country['cca2']}"
            elif not country_name and "cca3" in country:
                country_name = f"Country code: {country['cca3']}"
            elif not country_name:
                country_name = "Unnamed country"

            missed_countries[country_name] = "No match found"

    return missed_countries


def create_country_mappings(countries_data):
    """Create mappings for faster lookups by name, code, capital, and region"""
    name_to_index = {}
    code_to_index = {}
    capital_to_index = {}
    region_subregion_to_index = {}

    for i, country in enumerate(countries_data):
        # Map names
        if "name" in country:
            country_names = get_all_country_names(country)
            for name in country_names:
                if name:
                    name_to_index[name] = i

        # Map codes
        if "cca2" in country:
            code = country["cca2"].lower()
            code_to_index[code] = i
        if "cca3" in country:
            code = country["cca3"].lower()
            code_to_index[code] = i

        # Map capitals
        if "capital" in country and country["capital"]:
            for capital in country["capital"]:
                capital_normalized = normalize_country_name(capital.lower())
                capital_to_index[capital_normalized] = i

        # Map region/subregion
        if (
            "region" in country
            and "subregion" in country
            and country["region"]
            and country["subregion"]
        ):
            key = f"{country['region'].lower()}-{country['subregion'].lower()}"
            region_subregion_to_index[key] = i

    return name_to_index, code_to_index, capital_to_index, region_subregion_to_index
