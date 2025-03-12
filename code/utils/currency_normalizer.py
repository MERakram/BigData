import json
import re


def load_exchange_rates():
    """Load exchange rates from the local file"""
    try:
        rates_path = "../../data/input/exchage_rates.json"
        with open(rates_path, "r", encoding="utf-8") as f:
            exchange_data = json.load(f)

        if not exchange_data.get("success") or "rates" not in exchange_data:
            print("Warning: Exchange rates file may be invalid")
            return None

        # Verify the base is EUR
        if exchange_data.get("base") != "EUR":
            print(f"Warning: Expected EUR base but found {exchange_data.get('base')}")

        # Get rates directly (already EUR-based)
        eur_rates = exchange_data["rates"]

        # Calculate the inverse rates for conversion to EUR
        # If rate is 1.5 EUR to 1 X, then 1 X = 1/1.5 EUR
        inverse_rates = {currency: 1 / rate for currency, rate in eur_rates.items()}

        # Make sure EUR is exactly 1.0
        inverse_rates["EUR"] = 1.0

        print(
            f"Loaded {len(inverse_rates)} currency rates from local file (dated {exchange_data['date']})"
        )

        return {
            "rates": inverse_rates,
            "date": exchange_data["date"],
            "timestamp": exchange_data["timestamp"],
            "source": "local_file",
        }

    except Exception as e:
        print(f"Error loading exchange rates file: {str(e)}")
        return None


def normalize_currencies_to_eur(country_data, exchange_rates):
    """Normalize country's currency information to EUR"""

    # Add EUR currency conversion information
    if "currencies" in country_data and country_data["currencies"]:
        try:
            original_currency_code = list(country_data["currencies"].keys())[0]

            # Skip if already EUR
            if original_currency_code == "EUR":
                country_data["EUR_currency"] = {
                    "original_currency": country_data["currencies"]["EUR"],
                    "EUR_exchange_rate": 1.0,
                    "is_eur": True,
                }
            else:
                # Use local exchange rate if available
                if original_currency_code in exchange_rates["rates"]:
                    eur_rate = exchange_rates["rates"][original_currency_code]
                    country_data["EUR_currency"] = {
                        "original_currency": country_data["currencies"][
                            original_currency_code
                        ],
                        "original_code": original_currency_code,
                        "EUR_exchange_rate": eur_rate,
                        "is_eur": False,
                        "last_updated": exchange_rates["date"],
                    }
                else:
                    country_data["EUR_currency"] = {
                        "original_currency": country_data["currencies"][
                            original_currency_code
                        ],
                        "original_code": original_currency_code,
                        "error": f"Exchange rate not available for {original_currency_code}",
                    }
        except (IndexError, KeyError):
            country_data["EUR_currency"] = {"note": "Invalid currency structure"}
    else:
        country_data["EUR_currency"] = {"note": "No currency information available"}

    return country_data


def normalize_monetary_values(country_data):
    """Normalize any monetary values to EUR"""

    # Only proceed if we have a valid exchange rate
    if (
        "EUR_currency" not in country_data
        or "EUR_exchange_rate" not in country_data["EUR_currency"]
    ):
        return country_data

    exchange_rate = country_data["EUR_currency"].get("EUR_exchange_rate")

    if not exchange_rate or not isinstance(exchange_rate, (int, float)):
        return country_data

    # Normalize world_data monetary values if present
    if "world_data" in country_data:
        # Find monetary fields in world_data that might need conversion
        monetary_fields = [
            "GDP",
            "GDP per capita",
            "Minimum wage",
            "Average income",
            "Government expenditure",
            "External debt",
            "Tax revenue",
        ]

        for field in monetary_fields:
            if field in country_data["world_data"]:
                value = country_data["world_data"][field]
                # Check if it's a string that contains currency symbols or suffixes
                if isinstance(value, str):
                    # Remove currency symbols and commas
                    cleaned_value = re.sub(r"[$€£¥,]", "", value)
                    # Try to convert to float
                    try:
                        numeric_value = float(cleaned_value)
                        # Convert to EUR
                        if exchange_rate != 1.0:  # Skip if already in EUR
                            eur_value = numeric_value * exchange_rate
                            # Add EUR-normalized value but keep the original
                            country_data["world_data"][f"{field}_EUR"] = eur_value
                    except ValueError:
                        # If conversion fails, leave as is
                        pass

    # Similar normalization for happiness_data if present
    if "happiness_data" in country_data:
        # Process monetary fields in happiness_data
        monetary_fields = ["GDP_per_capita", "economic_support", "financial_security"]

        for field in monetary_fields:
            if field in country_data["happiness_data"]:
                value = country_data["happiness_data"][field]
                if isinstance(value, (int, float)):
                    if exchange_rate != 1.0:
                        eur_value = value * exchange_rate
                        country_data["happiness_data"][f"{field}_EUR"] = eur_value

    return country_data
