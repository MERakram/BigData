import re


def normalize_per_capita(country_data):
    """Calculate per-capita metrics where applicable"""
    population = None
    if "population" in country_data:
        population = country_data["population"]
    elif "world_data" in country_data and "Population" in country_data["world_data"]:
        # Handle population that might be formatted as string with commas
        pop_value = country_data["world_data"]["Population"]
        if isinstance(pop_value, str):
            pop_value = pop_value.replace(",", "").strip()
            try:
                population = int(pop_value)
            except ValueError:
                pass
        else:
            population = pop_value

    # Only calculate per capita if we have a valid population
    if population and isinstance(population, (int, float)) and population > 0:
        # Add general population data for easy reference
        country_data["normalized_data"] = country_data.get("normalized_data", {})
        country_data["normalized_data"]["population"] = population

        # Calculate per-capita metrics for world_data
        if "world_data" in country_data:
            # Common fields that make sense per-capita
            per_capita_fields = [
                "GDP",
                "Government expenditure",
                "External debt",
                "Tax revenue",
            ]

            for field in per_capita_fields:
                if field in country_data["world_data"]:
                    value = country_data["world_data"][field]
                    # Try to convert to numeric if it's a string
                    if isinstance(value, str):
                        value = re.sub(r"[$€£¥,]", "", value)
                        try:
                            value = float(value)
                        except ValueError:
                            continue

                    if isinstance(value, (int, float)):
                        per_capita = value / population
                        if "per_capita_metrics" not in country_data["normalized_data"]:
                            country_data["normalized_data"]["per_capita_metrics"] = {}
                        country_data["normalized_data"]["per_capita_metrics"][
                            f"{field}_per_capita"
                        ] = per_capita

    return country_data


def normalize_area_metrics(country_data):
    """Calculate area-based metrics (per sq km)"""
    area = None
    if "area" in country_data:
        area = country_data["area"]
    elif "world_data" in country_data and "Area" in country_data["world_data"]:
        area_value = country_data["world_data"]["Area"]
        if isinstance(area_value, str):
            area_value = area_value.replace(",", "").strip()
            try:
                area = float(area_value)
            except ValueError:
                pass
        else:
            area = area_value

    if area and isinstance(area, (int, float)) and area > 0:
        # Setup normalized_data if it doesn't exist
        country_data["normalized_data"] = country_data.get("normalized_data", {})
        country_data["normalized_data"]["area_sq_km"] = area

        # Calculate population density if population is available
        if "population" in country_data or (
            "normalized_data" in country_data
            and "population" in country_data["normalized_data"]
        ):
            population = country_data.get(
                "population", country_data.get("normalized_data", {}).get("population")
            )

            if population and isinstance(population, (int, float)):
                density = population / area
                country_data["normalized_data"]["population_density"] = density

    return country_data


def calculate_development_index(country_data):
    """Calculate a development index based on GDP, life expectancy, etc."""
    if "world_data" not in country_data:
        return country_data

    indicators = {}

    # Extract GDP per capita
    if "GDP per capita" in country_data.get("world_data", {}):
        gdp_val = country_data["world_data"]["GDP per capita"]
        if isinstance(gdp_val, str):
            # Clean numeric value
            gdp_val = re.sub(r"[$€£¥,]", "", gdp_val)
            try:
                indicators["gdp_pc"] = float(gdp_val)
            except ValueError:
                pass
        elif isinstance(gdp_val, (int, float)):
            indicators["gdp_pc"] = gdp_val

    # Extract life expectancy
    if "Life expectancy" in country_data.get("world_data", {}):
        life_val = country_data["world_data"]["Life expectancy"]
        if isinstance(life_val, str):
            # Extract just the number
            life_match = re.search(r"(\d+(\.\d+)?)", life_val)
            if life_match:
                indicators["life_exp"] = float(life_match.group(1))
        elif isinstance(life_val, (int, float)):
            indicators["life_exp"] = life_val

    # Extract literacy rate
    if "Literacy rate" in country_data.get("world_data", {}):
        lit_val = country_data["world_data"]["Literacy rate"]
        if isinstance(lit_val, str):
            # Remove % and convert to number
            lit_val = lit_val.replace("%", "").strip()
            try:
                indicators["literacy"] = float(lit_val)
            except ValueError:
                pass
        elif isinstance(lit_val, (int, float)):
            indicators["literacy"] = lit_val

    # Calculate index if we have at least two indicators
    if len(indicators) >= 2:
        # Create normalized values (0-1 scale)
        normalized = {}

        if "gdp_pc" in indicators:
            # Max GDP per capita capped at $100,000
            normalized["gdp_pc"] = min(indicators["gdp_pc"] / 100000, 1.0) * 0.4

        if "life_exp" in indicators:
            # Max life expectancy capped at 90 years
            normalized["life_exp"] = min(indicators["life_exp"] / 90, 1.0) * 0.4

        if "literacy" in indicators:
            # Already on 0-100 scale
            normalized["literacy"] = min(indicators["literacy"] / 100, 1.0) * 0.2

        # Calculate weighted average
        weights_sum = sum(
            [
                0.4 if "gdp_pc" in normalized else 0,
                0.4 if "life_exp" in normalized else 0,
                0.2 if "literacy" in normalized else 0,
            ]
        )

        if weights_sum > 0:
            dev_index = sum(normalized.values()) / weights_sum

            # Add to normalized data
            if "normalized_data" not in country_data:
                country_data["normalized_data"] = {}
            if "indices" not in country_data["normalized_data"]:
                country_data["normalized_data"]["indices"] = {}

            country_data["normalized_data"]["indices"]["development_index"] = round(
                dev_index, 3
            )
            country_data["normalized_data"]["indices"]["index_components"] = indicators

    return country_data
