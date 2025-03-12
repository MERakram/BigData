def filter_countries_with_data(data):
    """Filter countries to keep only those with happiness or world data"""
    filtered_countries = []
    total = len(data)
    filtered_out = 0
    removed_countries = []

    for country_data in data:
        country_name = country_data.get("name", {}).get("common", "Unknown")

        has_happiness = (
            "happiness_data" in country_data
            and len(country_data.get("happiness_data", {})) > 1
        )
        has_world_data = (
            "world_data" in country_data and len(country_data.get("world_data", {})) > 1
        )

        if has_happiness or has_world_data:
            filtered_countries.append(country_data)
        else:
            filtered_out += 1
            removed_countries.append(country_name)

    print(f"Filtered out {filtered_out} countries without happiness or world data")
    print(f"Kept {len(filtered_countries)} countries with relevant data")

    if removed_countries:
        print("\nRemoved countries (first 10):")
        for country in removed_countries[:10]:
            print(f"  - {country}")

        if len(removed_countries) > 10:
            print(f"  ... and {len(removed_countries) - 10} more")

    return filtered_countries
