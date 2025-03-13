"""
Comparison of Dimensionality Reduction Techniques
================================================

This script compares the performance and results of different dimensionality reduction
techniques (t-SNE and MDS) when applied to country data analysis.

The script:
1. Loads the analyzed country data
2. Applies both t-SNE and MDS for dimensionality reduction
3. Creates side-by-side visualizations for comparison
4. Compares performance metrics and cluster separation
5. Saves the comparison results and visualization
6. Generates a markdown report with analysis and recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import json
import sys
import datetime
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import silhouette_score

# Import from other files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsne_reduction import apply_tsne, load_data
from mds_reduction import apply_mds
from utils.reduction_comparison_report import generate_comparison_report


def compare_methods(
    df, features=["happiness_score", "gdp_per_capita_eur", "exchange_rate_eur"]
):
    """Compare t-SNE and MDS dimensionality reduction techniques

    Args:
        df (pandas.DataFrame): DataFrame containing country data
        features (list): List of features to use for dimensionality reduction

    Returns:
        tuple: (t-SNE results DataFrame, MDS results DataFrame, comparison metrics)
    """
    # Apply t-SNE
    tsne_df, tsne_metrics = apply_tsne(df, features)

    # Apply MDS
    mds_df, mds_metrics = apply_mds(df, features)

    # Combine metrics for comparison
    comparison = {"t-SNE": tsne_metrics, "MDS": mds_metrics, "features_used": features}

    return tsne_df, mds_df, comparison


def visualize_comparison(tsne_df, mds_df, comparison, output_dir="../../data/output"):
    """Create side-by-side visualization of t-SNE and MDS results

    Args:
        tsne_df (pandas.DataFrame): DataFrame with t-SNE results
        mds_df (pandas.DataFrame): DataFrame with MDS results
        comparison (dict): Comparison metrics
        output_dir (str): Directory to save visualization

    Returns:
        pandas.DataFrame: Comparison metrics as a DataFrame
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # t-SNE plot
    if "cluster" in tsne_df.columns:
        sns.scatterplot(
            data=tsne_df,
            x="tsne_x",
            y="tsne_y",
            hue="cluster",
            palette="viridis",
            size="population",
            sizes=(20, 200),
            alpha=0.6,
            ax=axes[0],
        )
    else:
        sns.scatterplot(
            data=tsne_df,
            x="tsne_x",
            y="tsne_y",
            size="population",
            sizes=(20, 200),
            alpha=0.6,
            ax=axes[0],
        )

    # Add labels for t-SNE
    top_countries_tsne = tsne_df.nlargest(10, "population")
    for idx, row in top_countries_tsne.iterrows():
        axes[0].text(
            row["tsne_x"] + 0.05,
            row["tsne_y"] + 0.05,
            row["country"],
            fontsize=9,
            alpha=0.7,
        )

    # MDS plot
    if "cluster" in mds_df.columns:
        sns.scatterplot(
            data=mds_df,
            x="mds_x",
            y="mds_y",
            hue="cluster",
            palette="viridis",
            size="population",
            sizes=(20, 200),
            alpha=0.6,
            ax=axes[1],
        )
    else:
        sns.scatterplot(
            data=mds_df,
            x="mds_x",
            y="mds_y",
            size="population",
            sizes=(20, 200),
            alpha=0.6,
            ax=axes[1],
        )

    # Add labels for MDS
    top_countries_mds = mds_df.nlargest(10, "population")
    for idx, row in top_countries_mds.iterrows():
        axes[1].text(
            row["mds_x"] + 0.05,
            row["mds_y"] + 0.05,
            row["country"],
            fontsize=9,
            alpha=0.7,
        )

    # Set titles and labels
    axes[0].set_title(
        f"t-SNE Visualization\nPerplexity: {comparison['t-SNE']['parameters']['perplexity']}"
    )
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")

    # Check if the MDS parameters include 'metric' key
    if "metric" in comparison["MDS"]["parameters"]:
        metric_type = (
            "Metric" if comparison["MDS"]["parameters"]["metric"] else "Non-metric"
        )
    else:
        metric_type = (
            "Metric"
            if comparison["MDS"]["parameters"].get("metric_bool", True)
            else "Non-metric"
        )

    axes[1].set_title(
        f"{metric_type} MDS Visualization\nStress: {comparison['MDS']['stress']:.4f}"
    )
    axes[1].set_xlabel("MDS Dimension 1")
    axes[1].set_ylabel("MDS Dimension 2")

    # Add annotations with metrics
    tsne_annotation = (
        f"Execution time: {comparison['t-SNE']['execution_time_seconds']:.2f}s\n"
        f"Samples: {comparison['t-SNE']['n_samples']}"
    )
    if comparison["t-SNE"]["cluster_separation"] is not None:
        tsne_annotation += (
            f"\nSilhouette score: {comparison['t-SNE']['cluster_separation']:.3f}"
        )

    axes[0].annotate(
        tsne_annotation,
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
    )

    mds_annotation = (
        f"Execution time: {comparison['MDS']['execution_time_seconds']:.2f}s\n"
        f"Stress: {comparison['MDS']['stress']:.4f}\n"
        f"Samples: {comparison['MDS']['n_samples']}"
    )
    if comparison["MDS"]["cluster_separation"] is not None:
        mds_annotation += (
            f"\nSilhouette score: {comparison['MDS']['cluster_separation']:.3f}"
        )

    axes[1].annotate(
        mds_annotation,
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
    )

    plt.suptitle("Comparison of Dimensionality Reduction Techniques", fontsize=16)
    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, "dim_reduction_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison visualization to {output_path}")

    # Create a summary table for the report
    method_comparison = pd.DataFrame(
        {
            "Method": ["t-SNE", "MDS"],
            "Execution Time (s)": [
                comparison["t-SNE"]["execution_time_seconds"],
                comparison["MDS"]["execution_time_seconds"],
            ],
            "Cluster Separation": [
                comparison["t-SNE"]["cluster_separation"],
                comparison["MDS"]["cluster_separation"],
            ],
        }
    )

    if "stress" in comparison["MDS"]:
        method_comparison.loc[1, "Stress"] = comparison["MDS"]["stress"]

    # Save comparison metrics
    comparison_path = os.path.join(output_dir, "dim_reduction_comparison.json")
    with open(comparison_path, "w") as f:
        # Convert numpy values to regular Python types for JSON serialization
        # Handle both float64 and float32 types (and any other numpy numeric types)
        json_serializable = {
            "t-SNE": {
                k: (
                    float(v)
                    if isinstance(v, (np.number, np.float32, np.float64))
                    else v
                )
                for k, v in comparison["t-SNE"].items()
            },
            "MDS": {
                k: (
                    float(v)
                    if isinstance(v, (np.number, np.float32, np.float64))
                    else v
                )
                for k, v in comparison["MDS"].items()
            },
            "features_used": comparison["features_used"],
        }
        json.dump(json_serializable, f, indent=4)

    print(f"Saved comparison metrics to {comparison_path}")

    # Generate markdown table for the report
    print("\nDimensionality Reduction Performance Comparison:")
    print(method_comparison.to_markdown(index=False))

    return method_comparison


def main():
    """Main function to compare dimensionality reduction techniques"""
    print("Starting comparison of dimensionality reduction techniques...")

    # Load data
    df = load_data()
    if df is None:
        return

    # Compare methods
    features = ["happiness_score", "gdp_per_capita_eur", "exchange_rate_eur"]
    tsne_df, mds_df, comparison = compare_methods(df, features)

    # Visualize comparison
    method_comparison = visualize_comparison(tsne_df, mds_df, comparison)

    # Generate comparison report with the correct parameters
    output_dir = "../../data/output"
    report_path = os.path.join(output_dir, "dim_reduction_report.md")

    try:
        # Generate report using the imported function - fixed function call
        generate_comparison_report(tsne_df, mds_df, comparison, output_dir=output_dir)
        print(f"Generated dimensionality reduction comparison report at {report_path}")
    except Exception as e:
        print(
            f"Failed to generate dimensionality reduction comparison report: {str(e)}"
        )

    # Determine which method performed better for cluster separation
    if (
        comparison["t-SNE"]["cluster_separation"] is not None
        and comparison["MDS"]["cluster_separation"] is not None
    ):
        if (
            comparison["t-SNE"]["cluster_separation"]
            > comparison["MDS"]["cluster_separation"]
        ):
            better_method = "t-SNE"
            score_diff = (
                comparison["t-SNE"]["cluster_separation"]
                - comparison["MDS"]["cluster_separation"]
            )
        else:
            better_method = "MDS"
            score_diff = (
                comparison["MDS"]["cluster_separation"]
                - comparison["t-SNE"]["cluster_separation"]
            )

        print(
            f"\nBased on cluster separation, {better_method} performed better by {score_diff:.3f} silhouette score."
        )

    # Compare execution times
    tsne_time = comparison["t-SNE"]["execution_time_seconds"]
    mds_time = comparison["MDS"]["execution_time_seconds"]

    if tsne_time < mds_time:
        faster_method = "t-SNE"
        time_diff = mds_time - tsne_time
    else:
        faster_method = "MDS"
        time_diff = tsne_time - mds_time

    print(
        f"In terms of execution time, {faster_method} was faster by {time_diff:.2f} seconds."
    )

    print("\nDimensionality reduction comparison complete!")


if __name__ == "__main__":
    main()
