import json
from datetime import datetime


def generate_countries_list_report():
    """Generate a simple MD report of countries that were kept vs filtered out"""
    try:
        # Load the original input data and final normalized/filtered data for comparison
        input_path = "../../data/output/enriched_countries_final.json"
        output_path = "../../data/output/normalized_countries.json"

        print("Generating countries list report...")

        with open(input_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)

        with open(output_path, "r", encoding="utf-8") as f:
            normalized_data = json.load(f)

        # Initialize lists for countries
        kept_countries = []
        filtered_out_countries = []

        # Get list of kept countries from the normalized output
        for country in normalized_data:
            country_name = country.get("name", {}).get("common", "Unknown")
            kept_countries.append(country_name)

        # Identify filtered out countries by comparing with original data
        for country in original_data:
            country_name = country.get("name", {}).get("common", "Unknown")
            if country_name not in kept_countries:
                filtered_out_countries.append(country_name)

        # Initialize the report content
        report = []
        report.append("# Countries Filtered Status Report")
        report.append("\nGenerated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        report.append("\n## Summary")
        report.append(
            "Countries were filtered based on having sufficient data in either:"
        )
        report.append("- `happiness_data` (happiness index metrics)")
        report.append("- `world_data` (economic and demographic indicators)")
        report.append(
            "\nAny country lacking both types of data was filtered out of the final dataset."
        )

        # Summary stats
        total_countries = len(original_data)
        kept_count = len(kept_countries)
        filtered_count = len(filtered_out_countries)

        report.append("\n## Statistics")
        report.append(f"- **Total countries:** {total_countries}")
        report.append(
            f"- **Countries kept:** {kept_count} ({kept_count/total_countries:.1%})"
        )
        report.append(
            f"- **Countries filtered out:** {filtered_count} ({filtered_count/total_countries:.1%})"
        )

        # List kept countries
        report.append("\n## Kept Countries")
        report.append(
            "These countries had sufficient happiness data and/or world data to be included in the final normalized dataset:\n"
        )

        for country in sorted(kept_countries):
            report.append(f"- {country}")

        # List filtered out countries
        report.append("\n## Filtered Out Countries")
        report.append(
            "These countries lacked sufficient happiness and world data and were therefore removed from the final dataset:\n"
        )

        for country in sorted(filtered_out_countries):
            report.append(f"- {country}")

        # Add data type statistics
        has_happiness_count = 0
        has_world_data_count = 0
        has_both_count = 0

        for country in normalized_data:
            has_happiness = (
                "happiness_data" in country
                and len(country.get("happiness_data", {})) > 1
            )
            has_world_data = (
                "world_data" in country and len(country.get("world_data", {})) > 1
            )

            if has_happiness and has_world_data:
                has_both_count += 1
            elif has_happiness:
                has_happiness_count += 1
            elif has_world_data:
                has_world_data_count += 1

        report.append("\n## Data Completeness")
        report.append("| Data Type | Number of Countries |")
        report.append("|-----------|---------------------|")
        report.append(f"| Both happiness and world data | {has_both_count} |")
        report.append(f"| Only happiness data | {has_happiness_count} |")
        report.append(f"| Only world data | {has_world_data_count} |")
        report.append(f"| Neither (filtered out) | {filtered_count} |")

        # Write the report to file
        report_path = "../../data/output/countries_normalized_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))

        print(f"Countries list report generated at {report_path}")
        return True

    except Exception as e:
        print(f"Error generating countries list report: {str(e)}")
        import traceback

        traceback.print_exc()
        return False
