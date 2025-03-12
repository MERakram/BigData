"""
Country Data Analysis Script
============================

This script analyzes the relationship between happiness, GDP per capita, and currency strength
across different countries to answer the research question:
"How do happiness, GDP, and currency strength relate to each other?"

The script:
1. Loads country data from normalized_countries.json
2. Extracts relevant metrics (happiness scores, GDP per capita, exchange rates)
3. Computes correlations between these variables
4. Performs K-means clustering to identify country groups
5. Generates visualizations and saves the results
6. Creates a comprehensive markdown report of the findings

Data source: normalized_countries.json containing country data with happiness scores,
economic indicators, and currency exchange rates relative to EUR.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import sys
import traceback
from datetime import datetime

# Add parent directory to path to allow importing utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_analysis_report import generate_markdown_report


def load_data():
    """Load the countries data"""
    try:
        # Define data file path
        data_path = "../../data/output/normalized_countries.json"

        if os.path.exists(data_path):
            print(f"Loading data from {data_path}...")
            with open(data_path, "r", encoding="utf-8") as f:
                countries_data = json.load(f)
            print(f"Loaded data for {len(countries_data)} countries")
            return countries_data, data_path
        else:
            print(f"Error: Could not find data file at {data_path}")
            print(
                "Please make sure the normalized_countries.json file exists in the data/output directory"
            )
            # If main file not found, try with enriched_countries_happiness.json
            alt_path = "../../data/output/enriched_countries_happiness.json"
            if os.path.exists(alt_path):
                print(
                    f"Found alternative data file at {alt_path}, trying to use that instead..."
                )
                with open(alt_path, "r", encoding="utf-8") as f:
                    countries_data = json.load(f)
                print(f"Loaded data for {len(countries_data)} countries")
                return countries_data, alt_path
            return None, None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        traceback.print_exc()
        return None, None


def extract_metrics(countries_data, file_path):
    """Extract happiness, GDP and currency metrics from the dataset"""
    metrics = []

    # Track sources for debugging
    happiness_source_paths = {}
    gdp_source_paths = {}
    currency_source_paths = {}

    for country in countries_data:
        try:
            if not isinstance(country, dict):
                continue

            # Get country name safely
            country_name = "Unknown"
            if "name" in country and isinstance(country["name"], dict):
                if "common" in country["name"]:
                    country_name = country["name"]["common"]

            # Extract happiness score
            happiness_score = None
            happiness_source = None

            # Check if happiness data exists in happiness_data.Score
            if "happiness_data" in country and isinstance(
                country["happiness_data"], dict
            ):
                if "Score" in country["happiness_data"]:
                    happiness_score = float(country["happiness_data"]["Score"])
                    happiness_source = "happiness_data.Score"
                elif "score" in country["happiness_data"]:
                    happiness_score = float(country["happiness_data"]["score"])
                    happiness_source = "happiness_data.score"

            # Extract GDP per capita
            gdp_per_capita = None
            gdp_source = None

            # Try normalized_data.per_capita_metrics.GDP_per_capita
            if "normalized_data" in country and isinstance(
                country["normalized_data"], dict
            ):
                if "per_capita_metrics" in country["normalized_data"] and isinstance(
                    country["normalized_data"]["per_capita_metrics"], dict
                ):
                    if (
                        "GDP_per_capita"
                        in country["normalized_data"]["per_capita_metrics"]
                    ):
                        gdp_per_capita = float(
                            country["normalized_data"]["per_capita_metrics"][
                                "GDP_per_capita"
                            ]
                        )
                        gdp_source = "normalized_data.per_capita_metrics.GDP_per_capita"

            # If not found, try extracting from world_data.GDP and population
            if (
                gdp_per_capita is None
                and "world_data" in country
                and isinstance(country["world_data"], dict)
            ):
                if (
                    "GDP" in country["world_data"]
                    and country["world_data"]["GDP"] is not None
                ):
                    # Clean the GDP string and convert to float
                    gdp_str = country["world_data"]["GDP"]
                    if isinstance(gdp_str, str):
                        gdp_float = float(gdp_str.replace("$", "").replace(",", ""))
                    else:
                        gdp_float = float(gdp_str)

                    # Get population
                    population_val = None
                    if "population" in country and country["population"] is not None:
                        population_val = float(country["population"])
                    elif (
                        "Population" in country["world_data"]
                        and country["world_data"]["Population"] is not None
                    ):
                        pop_str = country["world_data"]["Population"]
                        if isinstance(pop_str, str):
                            population_val = float(pop_str.replace(",", ""))
                        else:
                            population_val = float(pop_str)

                    if population_val is not None and population_val > 0:
                        gdp_per_capita = gdp_float / population_val
                        gdp_source = "calculated_from_world_data_GDP_and_population"

                # Try direct GDP_EUR from world_data
                if gdp_per_capita is None and "GDP_EUR" in country["world_data"]:
                    gdp_eur = country["world_data"]["GDP_EUR"]
                    population_val = country.get("population", None)
                    if population_val is not None and float(population_val) > 0:
                        gdp_per_capita = float(gdp_eur) / float(population_val)
                        gdp_source = "world_data.GDP_EUR_per_capita"

            # Extract exchange rate
            exchange_rate = None
            currency_code = None
            currency_source = None

            # Check EUR_currency
            if "EUR_currency" in country and country["EUR_currency"] is not None:
                if "EUR_exchange_rate" in country["EUR_currency"]:
                    exchange_rate = float(country["EUR_currency"]["EUR_exchange_rate"])
                    currency_source = "EUR_currency.EUR_exchange_rate"
                    if "original_code" in country["EUR_currency"]:
                        currency_code = country["EUR_currency"]["original_code"]

            # Extract population
            population = None
            if "population" in country and country["population"] is not None:
                population = float(country["population"])
            elif (
                "world_data" in country
                and "Population" in country["world_data"]
                and country["world_data"]["Population"] is not None
            ):
                pop_str = country["world_data"]["Population"]
                if isinstance(pop_str, str):
                    population = float(pop_str.replace(",", ""))
                else:
                    population = float(pop_str)

            # Add countries with at least happiness score or GDP data
            if happiness_score is not None or gdp_per_capita is not None:
                metrics.append(
                    {
                        "country": country_name,
                        "happiness_score": (
                            happiness_score if happiness_score is not None else np.nan
                        ),
                        "gdp_per_capita_eur": (
                            gdp_per_capita if gdp_per_capita is not None else np.nan
                        ),
                        "population": population if population is not None else np.nan,
                        "currency_code": currency_code,
                        "exchange_rate_eur": (
                            exchange_rate if exchange_rate is not None else np.nan
                        ),
                    }
                )

                # Track sources for debugging
                if happiness_source:
                    happiness_source_paths[happiness_source] = (
                        happiness_source_paths.get(happiness_source, 0) + 1
                    )
                if gdp_source:
                    gdp_source_paths[gdp_source] = (
                        gdp_source_paths.get(gdp_source, 0) + 1
                    )
                if currency_source:
                    currency_source_paths[currency_source] = (
                        currency_source_paths.get(currency_source, 0) + 1
                    )

        except Exception as e:
            print(f"Error processing country {country_name}: {str(e)}")
            traceback.print_exc()
            continue

    print(f"Extracted metrics for {len(metrics)} countries with at least some data")

    # Show where metrics were found
    print("\nHappiness data sources:")
    for path, count in sorted(
        happiness_source_paths.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {path}: {count} countries")

    print("\nGDP data sources:")
    for path, count in sorted(
        gdp_source_paths.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {path}: {count} countries")

    print("\nCurrency data sources:")
    for path, count in sorted(
        currency_source_paths.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {path}: {count} countries")

    return pd.DataFrame(metrics)


def analyze_data(df):
    """Perform basic analysis on the extracted metrics"""
    print("\nBasic statistics:")
    print(df.describe())

    # Calculate correlations
    corr_df = df[["happiness_score", "gdp_per_capita_eur", "exchange_rate_eur"]].corr()
    print("\nCorrelations:")
    print(corr_df)

    results = {
        "basic_stats": df.describe().to_dict(),
        "correlations": corr_df.to_dict(),
        "clusters": None,
        "cluster_centers": None,
        "cluster_counts": None,
    }

    # Only proceed with clustering if we have enough data
    if len(df) >= 3:
        # Prepare data for clustering
        features = ["happiness_score", "gdp_per_capita_eur", "exchange_rate_eur"]
        cluster_data = df[features].dropna()

        if len(cluster_data) >= 3:
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)

            # Apply K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            df.loc[cluster_data.index, "cluster"] = kmeans.fit_predict(scaled_data)

            # Store cluster information for report
            results["clusters"] = df[df["cluster"].notna()][
                [
                    "country",
                    "happiness_score",
                    "gdp_per_capita_eur",
                    "exchange_rate_eur",
                    "cluster",
                ]
            ].to_dict("records")
            results["cluster_centers"] = pd.DataFrame(
                scaler.inverse_transform(kmeans.cluster_centers_), columns=features
            ).to_dict("records")
            results["cluster_counts"] = df["cluster"].value_counts().to_dict()

            # Generate visualizations
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=df.dropna(
                    subset=["happiness_score", "gdp_per_capita_eur", "cluster"]
                ),
                x="gdp_per_capita_eur",
                y="happiness_score",
                hue="cluster",
                palette="viridis",
                size="population",
                sizes=(20, 200),
            )
            plt.title("Happiness Score vs GDP per Capita")
            plt.xlabel("GDP per Capita (EUR)")
            plt.ylabel("Happiness Score")

            # Save the plot
            plt.savefig("../../data/output/happiness_gdp_clusters.png")
            print("\nSaved visualization to happiness_gdp_clusters.png")

    return df, results


def main():
    """Main function to execute the analysis pipeline"""
    print("Starting analysis of happiness, GDP, and currency strength relationships...")

    # Load data
    countries_data, file_path = load_data()
    if not countries_data:
        return

    # Extract relevant metrics
    df = extract_metrics(countries_data, file_path)
    if df.empty:
        print("Error: Could not extract any metrics for analysis")
        return

    # Print data availability stats
    availability_stats = {}
    print("\nData availability:")
    for column in ["happiness_score", "gdp_per_capita_eur", "exchange_rate_eur"]:
        available = df[column].notna().sum()
        percentage = available / len(df) * 100
        print(f"  {column}: {available}/{len(df)} countries ({percentage:.1f}%)")

        # Store for report
        availability_stats[column] = {"count": int(available), "percentage": percentage}

    # Analyze the data
    analyzed_df, results = analyze_data(df)

    # Save the processed data
    analyzed_df.to_csv("../../data/output/analyzed_countries.csv", index=False)
    print("\nSaved processed data to analyzed_countries.csv")

    # Generate markdown report
    generate_markdown_report(analyzed_df, results, availability_stats)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
