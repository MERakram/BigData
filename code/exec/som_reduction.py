"""
SOM Dimensionality Reduction for Country Analysis
==================================================

This script applies SOM (Self-Organizing Map) to reduce
the dimensionality of country data features for visualization purposes.

SOM is a neural network–based dimensionality reduction technique that maps
high-dimensional data onto a 2D grid while preserving topological relationships.

The script:
1. Loads normalized country data and cluster information
2. Applies SOM to reduce the data to 2D
3. Visualizes the results with K-means clusters
4. Evaluates performance metrics
5. Saves the visualization and performance data
6. Generates an interpretive report of the findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import os
import time
import json
import sys

# Import MiniSom
try:
    from minisom import MiniSom
except ImportError:
    print("[ERROR] MiniSom is not installed. Please run 'pip install minisom' first.")
    sys.exit(1)

# Add parent directory to path to allow importing utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from utils.som_report import generate_som_report  # Ensure you have a corresponding report module

def load_data():
    """Load both normalized country data and cluster information"""
    try:
        # Load normalized data
        normalized_path = "../../data/output/normalization/normalized_countries.json"
        if not os.path.exists(normalized_path):
            print(f"Normalized data file not found: {normalized_path}")
            return None

        # Load JSON data
        with open(normalized_path, "r") as f:
            data = json.load(f)
            print(f"Loaded JSON normalized data with {len(data)} countries")

        # Extract country features directly from JSON structure
        countries = []
        for country in data:
            try:
                # Get country name
                if "name" in country and "common" in country["name"]:
                    country_name = country["name"]["common"]
                else:
                    continue

                # Extract happiness score
                happiness_score = None
                if "happiness_data" in country and "Score" in country["happiness_data"]:
                    happiness_score = country["happiness_data"]["Score"]

                # Extract GDP per capita
                gdp_per_capita = None
                if "gdp_per_capita" in country:
                    gdp_per_capita = country["gdp_per_capita"]

                # Extract exchange rate
                exchange_rate = None
                if ("EUR_currency" in country and 
                    "EUR_exchange_rate" in country["EUR_currency"]):
                    exchange_rate = country["EUR_currency"]["EUR_exchange_rate"]

                # Extract population
                population = None
                if "population" in country:
                    population = country["population"]

                countries.append({
                    "country": country_name,
                    "happiness_score": happiness_score,
                    "gdp_per_capita_eur": gdp_per_capita,
                    "exchange_rate_eur": exchange_rate,
                    "population": population,
                })
            except Exception as e:
                print(f"Error processing a country: {e}")
                continue

        normalized_df = pd.DataFrame(countries)

        # Load analyzed data with cluster information from CSV
        analyzed_path = "../../data/output/analysis/analyzed_countries.csv"
        if not os.path.exists(analyzed_path):
            print(f"Analyzed data with clusters not found: {analyzed_path}")
            return None

        analyzed_df = pd.read_csv(analyzed_path)
        print(f"Loaded analyzed data with {len(analyzed_df)} countries")

        # Extract cluster information
        cluster_info = analyzed_df[["country", "cluster"]].dropna(subset=["cluster"])
        print(f"Found cluster information for {len(cluster_info)} countries")

        # Merge normalized data with cluster info
        merged_df = pd.merge(normalized_df, cluster_info, on="country", how="left")
        print(f"Merged data has {len(merged_df)} countries")
        print(f"Merged cluster information for {merged_df['cluster'].notna().sum()} countries")

        # Diagnostic: report feature availability
        print("\nFeature availability in dataset:")
        for feature in ["happiness_score", "gdp_per_capita_eur", "exchange_rate_eur"]:
            available = merged_df[feature].notna().sum()
            print(f"  {feature}: {available}/{len(merged_df)} non-null values ({available/len(merged_df)*100:.1f}%)")

        # Check complete rows for all features
        complete_rows = merged_df.dropna(subset=["happiness_score", "gdp_per_capita_eur", "exchange_rate_eur"]).shape[0]
        print(f"  Complete rows with all features: {complete_rows}/{len(merged_df)} ({complete_rows/len(merged_df)*100:.1f}%)")

        # Sample a few rows for inspection
        print("\nSample data (first 3 rows):")
        print(merged_df[["country", "happiness_score", "gdp_per_capita_eur", "exchange_rate_eur"]].head(3))
        return merged_df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def apply_som(
    df,
    features=["happiness_score", "gdp_per_capita_eur", "exchange_rate_eur"],
    som_x=10,
    som_y=10,
    iterations=1000,
    sigma=1.0,
    learning_rate=0.5
):
    """Apply SOM dimensionality reduction

    Args:
        df (pandas.DataFrame): DataFrame containing country data
        features (list): List of features to use for SOM
        som_x (int): Number of cells in the SOM grid (width)
        som_y (int): Number of cells in the SOM grid (height)
        iterations (int): Number of training iterations
        sigma (float): Neighborhood function spread
        learning_rate (float): Initial learning rate

    Returns:
        tuple: (DataFrame with SOM results, performance metrics)
    """
    data_clean = df.dropna(subset=features)
    print(f"Applying SOM to {len(data_clean)} countries with {len(features)} features")
    if len(data_clean) < 2:
        print("ERROR: Not enough countries with complete data for SOM analysis")
        metrics = {
            "method": "SOM",
            "error": "Insufficient data",
            "parameters": {"grid_size": f"{som_x}x{som_y}", "iterations": iterations, "sigma": sigma, "learning_rate": learning_rate},
            "execution_time_seconds": 0,
            "cluster_separation": None,
            "n_samples": len(data_clean),
            "n_features": len(features),
            "features_used": features,
        }
        return pd.DataFrame(), metrics

    # Extract numeric array for chosen features
    X = data_clean[features].values.astype(float)

    # Apply log transformation to skewed features
    X_transformed = np.copy(X)
    if "gdp_per_capita_eur" in features:
        idx = features.index("gdp_per_capita_eur")
        X_transformed[:, idx] = np.log1p(X[:, idx])
    if "exchange_rate_eur" in features:
        idx = features.index("exchange_rate_eur")
        X_transformed[:, idx] = np.log1p(X[:, idx])

    # Normalize data with min-max scaling to [0, 1]
    mins = X_transformed.min(axis=0)
    maxs = X_transformed.max(axis=0)
    rng = maxs - mins
    rng[rng == 0] = 1e-9  # avoid division by zero
    X_norm = (X_transformed - mins) / rng

    start_time = time.time()
    # Initialize and train the SOM
    som = MiniSom(som_x, som_y, len(features), sigma=sigma, learning_rate=learning_rate, random_seed=42)
    som.random_weights_init(X_norm)
    som.train_random(X_norm, iterations, verbose=False)
    exec_time = time.time() - start_time

    # Map each sample to its Best Matching Unit (BMU)
    coords = [som.winner(row) for row in X_norm]
    result_df = data_clean.copy()
    result_df["som_xcoord"] = [c[0] for c in coords]
    result_df["som_ycoord"] = [c[1] for c in coords]

    # Compute silhouette score if cluster labels exist
    cluster_sep = None
    if "cluster" in result_df.columns and result_df["cluster"].notna().any():
        unique_clusters = result_df["cluster"].dropna().unique()
        if len(unique_clusters) > 1:
            cluster_sep = silhouette_score(result_df[["som_xcoord", "som_ycoord"]], result_df["cluster"])

    metrics = {
        "method": "SOM",
        "parameters": {"grid_size": f"{som_x}x{som_y}", "iterations": iterations, "sigma": sigma, "learning_rate": learning_rate},
        "execution_time_seconds": exec_time,
        "cluster_separation": cluster_sep,
        "n_samples": len(data_clean),
        "n_features": len(features),
        "features_used": features
    }
    return result_df, metrics


def visualize_som(result_df, metrics, output_dir="../../data/output"):
    """Visualize SOM results

    Args:
        result_df (pandas.DataFrame): DataFrame with SOM results
        metrics (dict): Performance metrics
        output_dir (str): Directory to save visualization
    """
    if result_df.empty:
        print("No data available for visualization")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 10))

    # Plot with cluster coloring if available
    if "cluster" in result_df.columns and result_df["cluster"].notna().any():
        plot_df = result_df.dropna(subset=["cluster"]).copy()
        plot_df["cluster"] = plot_df["cluster"].astype(int)
        sns.scatterplot(
            data=plot_df,
            x="som_xcoord",
            y="som_ycoord",
            hue="cluster",
            palette="viridis",
            size="population" if "population" in plot_df.columns else None,
            sizes=(20, 200),
            alpha=0.7
        )
    else:
        sns.scatterplot(
            data=result_df,
            x="som_xcoord",
            y="som_ycoord",
            size="population" if "population" in result_df.columns else None,
            sizes=(20, 200),
            alpha=0.7
        )

    # Annotate top countries by population (if available)
    if "population" in result_df.columns and result_df["population"].notna().any():
        top_countries = result_df.nlargest(15, "population")
    else:
        top_countries = result_df.head(15)
    for idx, row in top_countries.iterrows():
        plt.text(row["som_xcoord"]+0.1, row["som_ycoord"]+0.1, row["country"], fontsize=9, alpha=0.7)

    features_str = ", ".join(metrics.get("features_used", ["Unknown"]))
    plt.title(f"SOM Visualization ({metrics['parameters']['grid_size']} grid)\nFeatures: {features_str}")
    plt.xlabel("SOM X Coordinate")
    plt.ylabel("SOM Y Coordinate")

    annotation = f"Execution time: {metrics['execution_time_seconds']:.2f}s\nSamples: {metrics['n_samples']}"
    if metrics["cluster_separation"] is not None:
        annotation += f"\nSilhouette score: {metrics['cluster_separation']:.3f}"
    plt.annotate(annotation, xy=(0.02, 0.02), xycoords="axes fraction",
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

    output_path = os.path.join(output_dir, "som_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved SOM visualization to {output_path}")

    metrics_path = os.path.join(output_dir, "som_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved SOM metrics to {metrics_path}")


def main():
    print("Starting SOM dimensionality reduction analysis...\n")
    df = load_data()
    if df is None or df.empty:
        print("No data loaded. Exiting.")
        return

    features = ["happiness_score", "gdp_per_capita_eur", "exchange_rate_eur"]
    result_df, metrics = apply_som(df, features=features, som_x=10, som_y=10, iterations=1000, sigma=1.0, learning_rate=0.5)
    if result_df.empty:
        print("SOM analysis resulted in an empty dataset. Exiting.")
        return

    visualize_som(result_df, metrics)

    try:
        generate_som_report(result_df, metrics)
        print("SOM report generated successfully.")
    except Exception as e:
        print("Warning: SOM report generation failed.", e)

    print("\nSOM analysis complete!")
    print(f"Execution time: {metrics['execution_time_seconds']:.2f} seconds")
    if metrics["cluster_separation"] is not None:
        print(f"Silhouette score: {metrics['cluster_separation']:.3f}")


if __name__ == "__main__":
    main()
