import json
import time
from datetime import datetime
import sys
import os
import traceback

# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.currency_normalizer import (
    load_exchange_rates,
    normalize_currencies_to_eur,
    normalize_monetary_values,
)
from utils.metrics_normalizer import normalize_per_capita, normalize_area_metrics
from utils.text_normalizer import standardize_text_fields
from utils.filter_normalizer import filter_countries_with_data


def normalize_data(data):
    """Main function to apply all normalizations"""
    normalized_countries = []

    # Load exchange rates from the local file
    exchange_rates = load_exchange_rates()
    if not exchange_rates:
        print("ERROR: Could not load exchange rates, exiting...")
        return []

    total_countries = len(data)
    start_time = time.time()

    for i, country_data in enumerate(data):
        if not country_data:
            continue

        # Get country name for progress reporting
        country_name = country_data.get("name", {}).get("common", f"Country #{i+1}")

        # Show progress periodically
        if (i + 1) % 20 == 0 or i + 1 == 1 or i + 1 == total_countries:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            eta = avg_time * (total_countries - (i + 1))
            print(
                f"Processing {i+1}/{total_countries} - {country_name} (ETA: {eta:.1f}s)"
            )

        # Apply all normalizations
        country_data = normalize_currencies_to_eur(country_data, exchange_rates)
        country_data = normalize_monetary_values(country_data)
        country_data = normalize_per_capita(country_data)
        country_data = normalize_area_metrics(country_data)
        country_data = standardize_text_fields(country_data)

        # Add metadata about normalization
        country_data["metadata"] = {
            "normalized_on": datetime.now().strftime("%Y-%m-%d"),
            "normalization_version": "1.1",
            "exchange_rates_date": exchange_rates["date"],
            "exchange_rates_source": exchange_rates["source"],
            "base_currency": "EUR",
        }

        normalized_countries.append(country_data)

    # Print final stats
    total_time = time.time() - start_time
    print(
        f"Processed {len(normalized_countries)} countries in {total_time:.2f} seconds"
    )
    print(
        f"Average processing time: {total_time/len(normalized_countries):.4f} seconds per country"
    )

    # Filter countries to keep only those with relevant data
    filtered_countries = filter_countries_with_data(normalized_countries)

    # Save a temp copy of normalized data for report generation
    with open(
        "../../data/output/normalized_countries_all.json.tmp", "w", encoding="utf-8"
    ) as f:
        json.dump(normalized_countries, f, ensure_ascii=False, indent=2)

    return filtered_countries


def main():
    """Main function to load data, normalize it, and save results"""
    input_path = "../../data/output/enriched_countries_final.json"
    output_path = "../../data/output/normalized_countries.json"

    print(f"Loading data from {input_path}...")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Loaded {len(data)} countries. Starting normalization...")
        start_time = datetime.now()

        normalized_data = normalize_data(data)

        if normalized_data:
            # Save only the filtered data
            print(f"Saving normalized and filtered data to {output_path}...")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(normalized_data, f, ensure_ascii=False, indent=2)

            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"Normalization complete in {elapsed:.2f} seconds!")
            print(
                f"Saved {len(normalized_data)} filtered country records with EUR as base currency."
            )

            # Generate countries list report
            try:
                print("Generating countries list report...")
                # Import from utils directory
                from utils.normalization_report import generate_countries_list_report

                # Rename temporary file to expected path for report generation
                os.rename(
                    "../../data/output/normalized_countries_all.json.tmp",
                    "../../data/output/normalized_countries_all.json",
                )

                if generate_countries_list_report():
                    print("Countries list report generated successfully.")
                else:
                    print("Warning: Failed to generate countries list report.")

                # Clean up temporary file
                if os.path.exists("../../data/output/normalized_countries_all.json"):
                    os.remove("../../data/output/normalized_countries_all.json")

            except Exception as e:
                print(f"Warning: Could not generate countries list report: {str(e)}")
                traceback.print_exc()

                # Clean up temporary file on error
                if os.path.exists(
                    "../../data/output/normalized_countries_all.json.tmp"
                ):
                    os.remove("../../data/output/normalized_countries_all.json.tmp")
        else:
            print("ERROR: Normalization failed, no output generated.")

            # Clean up temporary file on error
            if os.path.exists("../../data/output/normalized_countries_all.json.tmp"):
                os.remove("../../data/output/normalized_countries_all.json.tmp")

    except Exception as e:
        print(f"Error in normalization process: {str(e)}")
        traceback.print_exc()

        # Clean up temporary file on error
        if os.path.exists("../../data/output/normalized_countries_all.json.tmp"):
            os.remove("../../data/output/normalized_countries_all.json.tmp")


if __name__ == "__main__":
    main()
