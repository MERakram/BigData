"""
t-SNE Report Generation Module
==============================

This module provides functions for generating detailed reports explaining t-SNE
dimensionality reduction results. The reports include analysis of cluster preservation,
performance metrics, and visual insights.
"""

import os
from datetime import datetime
import json
import pandas as pd
import numpy as np


def generate_tsne_report(result_df, metrics, output_dir="../../data/output"):
    """Generate a comprehensive markdown report of the t-SNE analysis results

    Args:
        result_df (pandas.DataFrame): DataFrame containing the t-SNE results
        metrics (dict): Dictionary containing t-SNE analysis metrics and parameters
        output_dir (str): Directory to save the report

    Returns:
        bool: True if report generation was successful, False otherwise
    """
    report_path = os.path.join(output_dir, "tsne_analysis_report.md")

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            # Report header
            f.write("# t-SNE Dimensionality Reduction Analysis Report\n\n")
            f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")

            # Introduction
            f.write("## 1. Introduction\n\n")
            f.write(
                "This report presents the results of applying t-SNE (t-Distributed Stochastic Neighbor Embedding) "
                "dimensionality reduction to country data. t-SNE is particularly effective at visualizing high-dimensional "
                "data by giving each datapoint a location in a two-dimensional map, while preserving local structure "
                "in the original data.\n\n"
            )

            # Methodology
            f.write("## 2. Methodology\n\n")
            features_used = metrics.get("features_used", ["Unknown features"])
            features_str = ", ".join(features_used)
            perplexity = metrics["parameters"].get("perplexity", "N/A")

            f.write(f"### 2.1 Features Used\n\n")
            f.write(f"The analysis was performed using the following features:\n\n")
            for feature in features_used:
                f.write(f"- **{feature}**\n")

            f.write("\n### 2.2 t-SNE Parameters\n\n")
            f.write(f"- **Perplexity**: {perplexity}\n")
            f.write("- **Iterations**: Default\n")
            f.write("- **Learning Rate**: Default\n\n")

            f.write(
                "*Note: Perplexity can be interpreted as a guess about the number of close "
                "neighbors each point has. The performance of t-SNE is fairly robust to changes "
                "in perplexity, with typical values between 5 and 50.*\n\n"
            )

            # Data overview
            f.write("## 3. Analysis Results\n\n")

            # Summary statistics
            f.write("### 3.1 Summary Statistics\n\n")
            f.write(f"- **Number of countries analyzed**: {metrics['n_samples']}\n")
            f.write(f"- **Number of features**: {metrics['n_features']}\n")
            f.write(
                f"- **Execution time**: {metrics['execution_time_seconds']:.2f} seconds\n"
            )

            # Cluster preservation
            f.write("\n### 3.2 Cluster Preservation\n\n")

            if metrics["cluster_separation"] is not None:
                silhouette = metrics["cluster_separation"]
                f.write(f"- **Silhouette score**: {silhouette:.3f}\n\n")

                if silhouette > 0.7:
                    quality = "excellent"
                    interpretation = (
                        "t-SNE has preserved the cluster structure extremely well. "
                        "The clusters are well-separated and cohesive in the reduced space."
                    )
                elif silhouette > 0.5:
                    quality = "good"
                    interpretation = (
                        "t-SNE has preserved the cluster structure well. "
                        "The clusters are reasonably separated in the reduced space."
                    )
                elif silhouette > 0.3:
                    quality = "fair"
                    interpretation = (
                        "t-SNE has preserved some cluster structure. "
                        "The clusters have moderate separation in the reduced space."
                    )
                else:
                    quality = "poor"
                    interpretation = (
                        "t-SNE has not preserved the cluster structure well. "
                        "The clusters are not well-separated in the reduced space."
                    )

                f.write(
                    f"The silhouette score indicates **{quality}** cluster preservation. {interpretation}\n\n"
                )

                # Count how many countries in each cluster
                if "cluster" in result_df.columns:
                    cluster_counts = result_df["cluster"].value_counts().sort_index()
                    f.write("#### Cluster Distribution in t-SNE Space\n\n")
                    f.write("| Cluster | Count | Percentage |\n")
                    f.write("|---------|-------|------------|\n")

                    for cluster, count in cluster_counts.items():
                        percentage = count / len(result_df) * 100
                        f.write(f"| {int(cluster)} | {count} | {percentage:.1f}% |\n")

                    f.write("\n")
            else:
                f.write("No cluster information was available for evaluation.\n\n")

            # Dimensionality reduction assessment
            f.write("### 3.3 t-SNE Visualization Assessment\n\n")

            f.write(
                "The t-SNE visualization has mapped the high-dimensional country data to a 2D space. "
                "This mapping preserves local structure, meaning that countries which are similar "
                "in terms of happiness, GDP, and currency strength are placed close together in the 2D map.\n\n"
            )

            f.write(
                "When examining the t-SNE visualization, consider the following:\n\n"
            )
            f.write(
                "1. **Clusters**: Countries forming distinct clusters may represent groups with similar economic and happiness characteristics.\n"
            )
            f.write(
                "2. **Outliers**: Countries appearing isolated in the visualization may have unique combinations of features.\n"
            )
            f.write(
                "3. **Gradients**: Gradual transitions between regions can indicate continuous relationships between economic factors and happiness.\n\n"
            )

            f.write(
                "*Note: t-SNE focuses on preserving local structure, so distances between well-separated clusters "
                "in the visualization are not necessarily meaningful. The global arrangement of clusters should not "
                "be over-interpreted.*\n\n"
            )

            # Interpretation of results
            f.write("## 4. Interpretation of Results\n\n")

            if (
                "happiness_score" in features_used
                and "gdp_per_capita_eur" in features_used
            ):
                f.write(
                    "### 4.1 Happiness and Economic Prosperity\n\n"
                    "The t-SNE visualization helps reveal patterns in how happiness scores relate "
                    "to economic indicators across countries. Countries with similar happiness levels and "
                    "economic characteristics tend to be positioned close to each other in the visualization.\n\n"
                    "Broadly, we can observe:\n\n"
                    "1. **Economic Similarity**: Countries with similar GDP per capita tend to form local neighborhoods, suggesting "
                    "that economic prosperity is a significant factor in the overall data structure.\n\n"
                    "2. **Happiness Clustering**: The colorization by clusters helps identify groups of countries that share "
                    "similar happiness characteristics despite potentially different economic situations.\n\n"
                )

            if "exchange_rate_eur" in features_used:
                f.write(
                    "### 4.2 Currency Strength Patterns\n\n"
                    "Currency exchange rates add an additional dimension to our understanding of economic wellbeing. "
                    "In the t-SNE visualization, countries with similar currency strength relative to the Euro may "
                    "be positioned near each other if this factor plays a significant role in the overall country profile.\n\n"
                    "The visualization may reveal whether currency strength correlates with other factors or forms its "
                    "own distinct patterns in the data structure.\n\n"
                )

            # View alongside cluster analysis
            f.write(
                "### 4.3 Comparison with K-means Clustering\n\n"
                "The t-SNE visualization is colored according to the K-means clusters previously identified. "
                "This allows us to see how well the dimensionality reduction has preserved the cluster structure.\n\n"
            )

            if metrics["cluster_separation"] is not None:
                silhouette = metrics["cluster_separation"]
                if silhouette > 0.5:
                    f.write(
                        "Based on the silhouette score, t-SNE has done a good job of preserving the clusters. "
                        "This suggests that the K-means clustering captured meaningful patterns in the data that "
                        "are also visible in the lower-dimensional representation.\n\n"
                    )
                else:
                    f.write(
                        "The moderate/low silhouette score suggests that some cluster structure may be lost in the "
                        "dimensionality reduction process. This could indicate that the clusters exist in dimensions "
                        "that t-SNE didn't prioritize in its non-linear mapping, or that the clusters themselves are "
                        "not strongly separated in the original space.\n\n"
                    )

            # Limitations
            f.write("## 5. Limitations\n\n")
            f.write(
                "When interpreting the t-SNE results, it's important to be aware of the following limitations:\n\n"
                "1. **Loss of Information**: The dimensionality reduction from multiple features to just two dimensions "
                "inevitably loses some information. Some patterns visible in the original high-dimensional space may not be visible "
                "in the t-SNE visualization.\n\n"
                "2. **Non-linear Mapping**: t-SNE performs a non-linear transformation that focuses on preserving local structure. "
                "This means that global patterns and distances between well-separated points are not accurately represented.\n\n"
                "3. **Random Initialization**: t-SNE involves random initialization, so running the algorithm multiple times "
                "can produce different visualizations. The general patterns should remain similar, but the specific layout may vary.\n\n"
                "4. **Parameter Sensitivity**: The perplexity parameter affects how t-SNE balances attention between local and global "
                "aspects of the data. Different perplexity values might highlight different patterns.\n\n"
            )

            # Conclusion
            f.write("## 6. Conclusion\n\n")

            features_list = []
            if "happiness_score" in features_used:
                features_list.append("happiness")
            if "gdp_per_capita_eur" in features_used:
                features_list.append("GDP per capita")
            if "exchange_rate_eur" in features_used:
                features_list.append("currency strength")

            features_text = ", ".join(features_list[:-1])
            if len(features_list) > 1:
                features_text += f" and {features_list[-1]}"
            else:
                features_text = features_list[0]

            f.write(
                f"The t-SNE analysis provides a valuable visualization of the relationships between {features_text} "
                f"across different countries. By reducing the dimensionality while preserving local structure, "
                f"t-SNE helps reveal patterns that might be difficult to detect in the original high-dimensional space.\n\n"
            )

            f.write(
                "The visualization complements the previous correlation and clustering analyses by providing "
                "a spatial representation of country similarities and differences. Countries that are positioned "
                "close together in the visualization generally have similar characteristics in terms of the selected features.\n\n"
            )

            f.write(
                "For further analysis, it would be valuable to compare these t-SNE results with those from other "
                "dimensionality reduction techniques, such as MDS (Multidimensional Scaling), to see if different "
                "methods highlight different aspects of the data structure.\n\n"
            )

            f.write("*End of Report*\n")

        print(f"t-SNE analysis report generated successfully: {report_path}")
        return True

    except Exception as e:
        print(f"Error generating t-SNE analysis report: {str(e)}")
        import traceback

        traceback.print_exc()
        return False
