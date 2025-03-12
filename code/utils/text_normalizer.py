import re


def standardize_text_fields(country_data):
    """Standardize text fields by removing HTML, extra whitespace, etc."""
    # Process description fields that might contain HTML or excess formatting
    if "unesco_sites" in country_data:
        for i, site in enumerate(country_data["unesco_sites"]):
            if "description" in site:
                # Remove HTML tags
                description = re.sub(r"<[^>]+>", "", site["description"])
                # Remove extra whitespace
                description = re.sub(r"\s+", " ", description).strip()
                country_data["unesco_sites"][i]["description"] = description

    # Clean up any world_data text fields
    if "world_data" in country_data:
        for field, value in country_data["world_data"].items():
            if isinstance(value, str):
                # Remove HTML and normalize whitespace
                cleaned_value = re.sub(r"<[^>]+>", "", value)
                cleaned_value = re.sub(r"\s+", " ", cleaned_value).strip()
                country_data["world_data"][field] = cleaned_value

    return country_data


def extract_numeric_value(text):
    """Extract a numeric value from text, handling common formats"""
    if not isinstance(text, str):
        return text

    # Remove currency symbols, commas, etc.
    cleaned = re.sub(r"[$€£¥,]", "", text)

    # Try to find a number in the text
    match = re.search(r"(-?\d+(\.\d+)?)", cleaned)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    return None
