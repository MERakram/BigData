"""
MDS Report Generator Module
===========================

This module generates an interpretive report for MDS dimensionality reduction
results, helping to understand patterns and relationships in country data.
"""

import pandas as pd
import os
import json
from datetime import datetime


def generate_mds_report(result_df, metrics, output_dir="../../data/output"):
    """Generate a comprehensive markdown report on MDS analysis results

    Args:
        result_df (pandas.DataFrame): DataFrame with MDS results
        metrics (dict): Performance metrics from MDS analysis
        output_dir (str): Directory to save the report

    Returns:
        bool: True if report was generated successfully, False otherwise
    """
    if result_df.empty:
        print("Cannot generate report: No data available")
        return False

    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Report path
        report_path = os.path.join(output_dir, "mds_analysis_report.md")

        # Extract key metrics
        method = metrics.get("method", "MDS")
        metric_type = (
            "Metric"
            if metrics.get("parameters", {}).get("metric", True)
            else "Non-metric"
        )
        execution_time = metrics.get("execution_time_seconds", "N/A")
        features = metrics.get("features_used", [])
        n_samples = metrics.get("n_samples", 0)
        n_features = metrics.get("n_features", 0)
        stress = metrics.get("stress", "N/A")
        cluster_separation = metrics.get("cluster_separation", None)

        # Calculate cluster statistics if available
        cluster_stats = None
        if "cluster" in result_df.columns and result_df["cluster"].notna().any():
            clustered_df = result_df.dropna(subset=["cluster"])
            cluster_stats = clustered_df.groupby("cluster").agg(
                {
                    "mds_x": ["mean", "min", "max"],
                    "mds_y": ["mean", "min", "max"],
                    "happiness_score": (
                        ["mean", "min", "max", "count"]
                        if "happiness_score" in clustered_df.columns
                        else []
                    ),
                    "gdp_per_capita_eur": (
                        ["mean", "min", "max"]
                        if "gdp_per_capita_eur" in clustered_df.columns
                        else []
                    ),
                    "exchange_rate_eur": (
                        ["mean", "min", "max"]
                        if "exchange_rate_eur" in clustered_df.columns
                        else []
                    ),
                }
            )

        # Generate the report
        with open(report_path, "w") as f:
            # Report header
            f.write(f"# {metric_type} MDS Dimensionality Reduction Analysis Report\n\n")
            f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")

            # Introduction
            f.write("## 1. Introduction\n\n")
            f.write(
                "This report presents the results of applying Multi-Dimensional Scaling (MDS) "
            )
            f.write(
                "to country data. MDS is a technique that preserves pairwise distances between data points "
            )
            f.write(
                "when reducing dimensionality, making it useful for understanding global structure and "
            )
            f.write(
                "visualizing relationships between countries based on economic and well-being indicators.\n\n"
            )

            # Methodology
            f.write("## 2. Methodology\n\n")
            f.write(
                f"The {metric_type} MDS algorithm was applied to {n_samples} countries using the following features:\n\n"
            )
            for feature in features:
                f.write(f"- **{feature}**\n")
            f.write("\n")

            f.write(f"**Parameters:**\n\n")
            f.write(f"- **Type**: {metric_type} MDS\n")
            f.write(f"- **Random state**: 42 (for reproducibility)\n\n")

            # Execution Statistics
            f.write("## 3. Execution Statistics\n\n")
            f.write(f"- **Runtime**: {execution_time:.2f} seconds\n")
            f.write(f"- **Number of countries analyzed**: {n_samples}\n")
            f.write(f"- **Number of features**: {n_features}\n")

            if isinstance(stress, (int, float)):
                f.write(f"- **Stress**: {stress:.4f}\n")
                f.write(
                    "  - Lower stress values indicate better preservation of distances\n"
                )

            if cluster_separation is not None:
                f.write(
                    f"- **Cluster separation (silhouette score)**: {cluster_separation:.3f}\n"
                )
                f.write(
                    "  - A higher silhouette score (closer to 1.0) indicates better-defined clusters\n"
                )
            f.write("\n")

            # Results Interpretation
            f.write("## 4. Results Interpretation\n\n")

            # Overall patterns
            f.write("### 4.1 Overall Patterns\n\n")
            f.write(
                "The MDS visualization reveals several key patterns in the country data:\n\n"
            )

            f.write(
                "1. **Distance Preservation**: MDS preserves distances between data points, "
            )
            f.write(
                "meaning countries that are dissimilar in the original feature space (happiness, GDP, and exchange rate) "
            )
            f.write("appear farther apart in the 2D visualization.\n\n")

            f.write(
                "2. **Global Structure**: Unlike t-SNE, MDS focuses on preserving the global structure of the data, "
            )
            f.write(
                "making it valuable for understanding overall patterns and relationships between all countries.\n\n"
            )

            # Stress interpretation
            if isinstance(stress, (int, float)):
                f.write("3. **Dimensionality Reduction Quality**: ")

                if stress < 0.05:
                    f.write(
                        "The very low stress value indicates an excellent representation of the original data structure "
                    )
                    f.write("in the reduced 2D space.\n\n")
                elif stress < 0.1:
                    f.write(
                        "The low stress value indicates a good representation of the original data structure "
                    )
                    f.write("in the reduced 2D space.\n\n")
                elif stress < 0.2:
                    f.write(
                        "The moderate stress value suggests a fair representation of the original data structure, "
                    )
                    f.write("though some distortion may be present.\n\n")
                else:
                    f.write(
                        "The relatively high stress value indicates that significant distortion was introduced "
                    )
                    f.write(
                        "when reducing the data to 2D, suggesting caution when interpreting fine details.\n\n"
                    )

            # Cluster analysis
            if cluster_stats is not None:
                f.write("### 4.2 Cluster Analysis in MDS Space\n\n")
                f.write(
                    "The following statistics describe each cluster in the reduced MDS space:\n\n"
                )

                for cluster_id in sorted(cluster_stats.index):
                    count = (
                        cluster_stats.loc[cluster_id][("happiness_score", "count")]
                        if ("happiness_score", "count") in cluster_stats.columns
                        else "N/A"
                    )
                    f.write(f"**Cluster {int(cluster_id)}** ({count} countries):\n\n")

                    # Get some representative countries from this cluster
                    if "cluster" in result_df.columns:
                        countries = result_df[
                            result_df["cluster"] == cluster_id
                        ].nlargest(
                            3,
                            (
                                "population"
                                if "population" in result_df.columns
                                else "mds_x"
                            ),
                        )
                        if not countries.empty:
                            country_list = ", ".join(countries["country"].tolist())
                            f.write(f"- **Representative countries**: {country_list}\n")

                    # Calculate statistics for happiness, GDP, etc. if available
                    if "happiness_score" in result_df.columns:
                        avg_happiness = cluster_stats.loc[cluster_id][
                            ("happiness_score", "mean")
                        ]
                        f.write(f"- **Average happiness score**: {avg_happiness:.2f}\n")

                    if "gdp_per_capita_eur" in result_df.columns:
                        avg_gdp = cluster_stats.loc[cluster_id][
                            ("gdp_per_capita_eur", "mean")
                        ]
                        f.write(f"- **Average GDP per capita (EUR)**: {avg_gdp:,.2f}\n")

                    if "exchange_rate_eur" in result_df.columns:
                        avg_er = cluster_stats.loc[cluster_id][
                            ("exchange_rate_eur", "mean")
                        ]
                        f.write(f"- **Average exchange rate to EUR**: {avg_er:.4f}\n")

                    # Position in MDS space
                    x_mean = cluster_stats.loc[cluster_id][("mds_x", "mean")]
                    y_mean = cluster_stats.loc[cluster_id][("mds_y", "mean")]
                    f.write(
                        f"- **Position in MDS space**: ({x_mean:.2f}, {y_mean:.2f})\n\n"
                    )

            # Limitations
            f.write("### 4.3 Limitations\n\n")
            f.write(
                "When interpreting MDS visualizations, it's important to consider the following limitations:\n\n"
            )

            f.write(
                "1. **Distance Interpretation**: MDS aims to preserve distances, but perfect preservation "
            )
            f.write(
                "in lower dimensions isn't always possible, especially when reducing from many features to just 2D.\n\n"
            )

            f.write(
                "2. **Stress Value Importance**: Higher stress values indicate greater distortion in the visualization. "
            )
            f.write(
                f"This analysis has a stress value of {stress if isinstance(stress, (int, float)) else 'N/A'}.\n\n"
            )

            f.write(
                "3. **Linear vs. Non-linear**: Metric MDS assumes linear relationships between features, "
            )
            f.write("which may not always hold for complex socioeconomic data.\n\n")

            # Conclusion
            f.write("## 5. Conclusion\n\n")
            f.write(
                "The MDS dimensionality reduction visualizes the global relationships between countries "
            )
            f.write(
                "based on their happiness scores, GDP per capita, and currency exchange rates. "
            )

            if cluster_separation and cluster_separation > 0.4:
                f.write(
                    "The visualization effectively preserves the cluster structure identified in the K-means analysis, "
                )
            else:
                f.write(
                    "The visualization shows the overall structure of the country data, though some cluster boundaries may be less distinct, "
                )

            f.write(
                "providing an intuitive way to understand the relationships between different countries.\n\n"
            )

            f.write(
                "MDS is particularly valuable for this dataset because it focuses on preserving the distances between all countries, "
            )
            f.write(
                "giving a reliable global view of how countries relate to each other in terms of economic and well-being metrics. "
            )
            f.write(
                "This global perspective complements the more locally-focused t-SNE approach, "
            )
            f.write(
                "and together they provide a comprehensive understanding of the data structure.\n\n"
            )

            f.write("*End of MDS Analysis Report*\n")

        print(f"MDS report generated successfully: {report_path}")
        return True

    except Exception as e:
        print(f"Error generating MDS report: {str(e)}")
        import traceback

        traceback.print_exc()
        return False
