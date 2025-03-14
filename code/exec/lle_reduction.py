"""
LLE Dimensionality Reduction for Country Analysis
==================================================

This script applies LLE (Locally Linear Embedding) to reduce
the dimensionality of country data features for visualization purposes.

LLE is a nonlinear dimensionality reduction technique that preserves 
the local neighborhood structure of high-dimensional data. It works by
representing each data point as a linear combination of its neighbors.

The script:
1. Loads normalized country data and cluster information
2. Applies LLE to reduce the data to 2D
3. Visualizes the results with K-means clusters
4. Evaluates performance metrics
5. Saves the visualization and performance data
6. Generates an interpretive report of the findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import silhouette_score
import os
import time
import json
import sys

# Add parent directory to path to allow importing utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tsne_report import generate_tsne_report


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
                    # Skip if no valid name
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
                if (
                    "EUR_currency" in country
                    and "EUR_exchange_rate" in country["EUR_currency"]
                ):
                    exchange_rate = country["EUR_currency"]["EUR_exchange_rate"]

                # Extract population
                population = None
                if "population" in country:
                    population = country["population"]

                # Create country entry
                countries.append(
                    {
                        "country": country_name,
                        "happiness_score": happiness_score,
                        "gdp_per_capita_eur": gdp_per_capita,
                        "exchange_rate_eur": exchange_rate,
                        "population": population,
                    }
                )
            except Exception as e:
                print(f"Error processing a country: {e}")
                continue

        # Convert to DataFrame
        normalized_df = pd.DataFrame(countries)

        # Load analyzed data with cluster information from CSV
        analyzed_path = "../../data/output/analysis/analyzed_countries.csv"
        if not os.path.exists(analyzed_path):
            print(f"Analyzed data with clusters not found: {analyzed_path}")
            return None

        # Load analyzed data to get cluster information
        analyzed_df = pd.read_csv(analyzed_path)
        print(f"Loaded analyzed data with {len(analyzed_df)} countries")

        # Extract cluster information from analyzed data
        cluster_info = analyzed_df[["country", "cluster"]].dropna(subset=["cluster"])
        print(f"Found cluster information for {len(cluster_info)} countries")

        # Merge normalized data with cluster information
        merged_df = pd.merge(normalized_df, cluster_info, on="country", how="left")

        print(f"Merged data has {len(merged_df)} countries")
        print(
            f"Merged cluster information for {merged_df['cluster'].notna().sum()} countries"
        )

        # Add diagnostic information to help debug data issues
        print("\nFeature availability in dataset:")
        for feature in ["happiness_score", "gdp_per_capita_eur", "exchange_rate_eur"]:
            available = merged_df[feature].notna().sum()
            print(
                f"  {feature}: {available}/{len(merged_df)} non-null values ({available/len(merged_df)*100:.1f}%)"
            )

        # Check how many countries have all three features
        complete_rows = merged_df.dropna(
            subset=["happiness_score", "gdp_per_capita_eur", "exchange_rate_eur"]
        ).shape[0]
        print(
            f"  Complete rows with all features: {complete_rows}/{len(merged_df)} ({complete_rows/len(merged_df)*100:.1f}%)"
        )

        # Check feature pairs if needed
        if complete_rows < 10:
            print("\nChecking feature pairs availability:")
            pairs = [
                ["happiness_score", "gdp_per_capita_eur"],
                ["happiness_score", "exchange_rate_eur"],
                ["gdp_per_capita_eur", "exchange_rate_eur"],
            ]
            for pair in pairs:
                pair_count = merged_df.dropna(subset=pair).shape[0]
                print(
                    f"  {pair[0]} and {pair[1]}: {pair_count}/{len(merged_df)} ({pair_count/len(merged_df)*100:.1f}%)"
                )

        # Sample the first few rows to inspect
        print("\nSample data (first 3 rows):")
        print(
            merged_df[
                [
                    "country",
                    "happiness_score",
                    "gdp_per_capita_eur",
                    "exchange_rate_eur",
                ]
            ].head(3)
        )

        return merged_df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def apply_lle(
    df,
    features=["happiness_score", "gdp_per_capita_eur", "exchange_rate_eur"],
    n_neighbors=15,
    method='standard',
):
    """Apply LLE dimensionality reduction

    Args:
        df (pandas.DataFrame): DataFrame containing country data
        features (list): List of features to use for dimensionality reduction
        n_neighbors (int): Number of neighbors to consider for manifold learning
        method (str): The method used for LLE ('standard', 'hessian', 'modified', or 'ltsa')

    Returns:
        tuple: (DataFrame with LLE results, performance metrics)
    """
    # Drop rows with NaN values in the features
    data_clean = df.dropna(subset=features)

    print(
        f"Applying LLE to {len(data_clean)} countries with {len(features)} features"
    )

    # If we have too few samples, try with fewer features or adjust n_neighbors
    if len(data_clean) < 10:
        print(f"WARNING: Only {len(data_clean)} complete samples available for LLE")

        # Try with two features if all three aren't available
        if len(features) > 2 and len(data_clean) < 3:
            best_pair = None
            max_samples = 0

            # Find the pair of features with the most complete data
            feature_pairs = []
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    pair = [features[i], features[j]]
                    pair_count = df.dropna(subset=pair).shape[0]
                    feature_pairs.append((pair, pair_count))
                    if pair_count > max_samples:
                        max_samples = pair_count
                        best_pair = pair

            if best_pair and max_samples > 10:
                print(
                    f"Switching to feature pair with most data: {best_pair} ({max_samples} samples)"
                )
                features = best_pair
                data_clean = df.dropna(subset=features)
                print(
                    f"Now using {len(data_clean)} countries with {len(features)} features"
                )

    # Exit gracefully if we still don't have enough data
    if len(data_clean) < 3:
        print("ERROR: Not enough countries with complete data for LLE analysis")
        empty_metrics = {
            "method": "LLE",
            "error": "Insufficient data",
            "parameters": {"n_neighbors": n_neighbors, "method": method},
            "execution_time_seconds": 0,
            "cluster_separation": None,
            "n_samples": len(data_clean),
            "n_features": len(features),
            "features_used": features,
        }
        return pd.DataFrame(), empty_metrics

    # Adjust n_neighbors if needed
    if n_neighbors >= len(data_clean):
        old_n_neighbors = n_neighbors
        n_neighbors = max(2, min(15, len(data_clean) // 2))
        print(
            f"Adjusting n_neighbors from {old_n_neighbors} to {n_neighbors} (must be less than sample count)"
        )

    # Select only numeric features for LLE
    X = data_clean[features].values

    # Apply log transformation to highly skewed features like GDP and exchange rate
    X_transformed = np.copy(X)

    # Handle log transformations if those features are present
    if "gdp_per_capita_eur" in features:
        idx = features.index("gdp_per_capita_eur")
        X_transformed[:, idx] = np.log1p(X[:, idx])
    if "exchange_rate_eur" in features:
        idx = features.index("exchange_rate_eur")
        X_transformed[:, idx] = np.log1p(X[:, idx])

    # Track performance
    start_time = time.time()

    # Apply LLE
    lle = LocallyLinearEmbedding(
        n_components=2,
        n_neighbors=n_neighbors,
        method=method,
        random_state=42
    )
    lle_results = lle.fit_transform(X_transformed)

    execution_time = time.time() - start_time

    # Store results in dataframe
    result_df = data_clean.copy()
    result_df["lle_x"] = lle_results[:, 0]
    result_df["lle_y"] = lle_results[:, 1]

    # Measure cluster separation using silhouette score if clusters exist
    cluster_separation = None
    if (
        "cluster" in result_df.columns
        and len(np.unique(result_df["cluster"].dropna())) > 1
    ):
        # Get only rows with cluster assignments
        clustered_data = result_df.dropna(subset=["cluster"])
        if len(clustered_data) >= 2:
            cluster_separation = silhouette_score(
                clustered_data[["lle_x", "lle_y"]],
                clustered_data["cluster"],
                random_state=42,
            )

    # Collect performance metrics
    metrics = {
        "method": "LLE",
        "parameters": {"n_neighbors": n_neighbors, "method": method},
        "reconstruction_error": lle.reconstruction_error_,
        "execution_time_seconds": execution_time,
        "cluster_separation": cluster_separation,
        "n_samples": len(data_clean),
        "n_features": len(features),
        "features_used": features,
    }

    return result_df, metrics


def visualize_lle(result_df, metrics, output_dir="../../data/output"):
    """Visualize LLE results

    Args:
        result_df (pandas.DataFrame): DataFrame with LLE results
        metrics (dict): Performance metrics
        output_dir (str): Directory to save visualization
    """
    # Handle empty results
    if result_df.empty:
        print("No data available for visualization")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 10))

    # Plot with cluster coloring if available
    if "cluster" in result_df.columns and result_df["cluster"].notna().any():
        # Filter out NaN clusters for visualization
        plot_df = result_df.dropna(subset=["cluster"]).copy()
        plot_df["cluster"] = plot_df["cluster"].astype(int)
        scatter = sns.scatterplot(
            data=plot_df,
            x="lle_x",
            y="lle_y",
            hue="cluster",
            palette="viridis",
            size="population" if "population" in plot_df.columns else None,
            sizes=(20, 200),
            alpha=0.6,
        )
    else:
        scatter = sns.scatterplot(
            data=result_df,
            x="lle_x",
            y="lle_y",
            size="population" if "population" in result_df.columns else None,
            sizes=(20, 200),
            alpha=0.6,
        )

    # Add country labels for prominent countries
    if "population" in result_df.columns and result_df["population"].notna().any():
        top_countries = result_df.nlargest(15, "population")
    else:
        top_countries = result_df.head(15)

    for idx, row in top_countries.iterrows():
        plt.text(
            row["lle_x"] + 0.05,
            row["lle_y"] + 0.05,
            row["country"],
            fontsize=9,
            alpha=0.7,
        )

    features_str = ", ".join(metrics.get("features_used", ["Unknown"]))
    method_str = metrics["parameters"]["method"].capitalize()
    plt.title(
        f'LLE ({method_str}) Visualization of Country Data\nFeatures: {features_str}\n'
        f'n_neighbors: {metrics["parameters"]["n_neighbors"]}'
    )
    plt.xlabel("LLE Dimension 1")
    plt.ylabel("LLE Dimension 2")

    # Add metrics as annotation
    annotation = (
        f"Execution time: {metrics['execution_time_seconds']:.2f}s\n"
        f"Samples: {metrics['n_samples']}\n"
        f"Reconstruction error: {metrics.get('reconstruction_error', 0):.4f}"
    )
    if metrics["cluster_separation"] is not None:
        annotation += f"\nSilhouette score: {metrics['cluster_separation']:.3f}"

    plt.annotate(
        annotation,
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
    )

    # Save plot
    output_path = os.path.join(output_dir, f"lle_{metrics['parameters']['method']}_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved LLE visualization to {output_path}")

    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, f"lle_{metrics['parameters']['method']}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved LLE metrics to {metrics_path}")


def main():
    print("Starting LLE dimensionality reduction analysis...\n")
    df = load_data()
    if df is None or df.empty:
        print("No data loaded. Exiting.")
        return

    features = ["happiness_score", "gdp_per_capita_eur", "exchange_rate_eur"]
    result_df, metrics = apply_lle(df, features=features, n_neighbors=15, method='standard')
    if result_df.empty:
        print("LLE analysis resulted in an empty dataset. Exiting.")
        return

    visualize_lle(result_df, metrics)

    try:
        generate_lle_report(result_df, metrics)
        print("LLE report generated successfully.")
    except Exception as e:
        print("Warning: LLE report generation failed.", e)

    print("\nLLE analysis complete!")
    print(f"Execution time: {metrics['execution_time_seconds']:.2f} seconds")
    if metrics["cluster_separation"] is not None:
        print(f"Silhouette score: {metrics['cluster_separation']:.3f}")


if __name__ == "__main__":
    main()
