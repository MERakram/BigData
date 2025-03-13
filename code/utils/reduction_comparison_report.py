"""
Dimensionality Reduction Comparison Report Generator
===================================================

This module provides functions for generating detailed comparison reports between
different dimensionality reduction techniques. The reports include comparative analysis,
performance metrics, visualization insights, and recommendations.
"""

import os
from datetime import datetime
import json
import pandas as pd
import numpy as np


def generate_comparison_report(
    tsne_df, mds_df, comparison, output_dir="../../data/output"
):
    """Generate a markdown report comparing t-SNE and MDS dimensionality reduction results

    Args:
        tsne_df (pandas.DataFrame): DataFrame with t-SNE results
        mds_df (pandas.DataFrame): DataFrame with MDS results
        comparison (dict): Dictionary with comparison metrics
        output_dir (str): Directory to save the report

    Returns:
        bool: True if report generation was successful, False otherwise
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "dimensionality_reduction_comparison.md")

    try:
        with open(report_path, "w") as f:
            # Report header
            f.write("# Dimensionality Reduction Techniques Comparison Report\n\n")
            f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")

            # Introduction
            f.write("## 1. Introduction\n\n")
            f.write(
                "This report presents a comprehensive comparison between two dimensionality reduction techniques: "
                "t-SNE (t-Distributed Stochastic Neighbor Embedding) and MDS (Multidimensional Scaling). "
                "Both techniques are used to visualize high-dimensional data in a lower-dimensional space, "
                "but they differ in their approaches, performance characteristics, and results.\n\n"
            )

            # Methodology
            f.write("## 2. Methodology\n\n")
            features_used = comparison.get("features_used", ["Unknown features"])
            features_str = ", ".join(features_used)

            f.write("### 2.1 Features Used\n\n")
            f.write("The comparison was performed using the following features:\n\n")
            for feature in features_used:
                f.write(f"- **{feature}**\n")
            f.write("\n")

            # t-SNE parameters
            f.write("### 2.2 t-SNE Parameters\n\n")
            perplexity = comparison["t-SNE"]["parameters"].get("perplexity", "N/A")
            f.write(f"- **Perplexity**: {perplexity}\n")
            f.write("- **Random State**: 42 (for reproducibility)\n\n")

            f.write(
                "*Note: Perplexity in t-SNE can be interpreted as a guess about the number of close neighbors each point has. "
                "The performance is fairly robust to changes in perplexity, with typical values between 5 and 50.*\n\n"
            )

            # MDS parameters
            f.write("### 2.3 MDS Parameters\n\n")
            metric = comparison["MDS"]["parameters"].get("metric", True)
            metric_str = "Metric" if metric else "Non-metric"
            f.write(f"- **Type**: {metric_str}\n")
            f.write("- **Random State**: 42 (for reproducibility)\n\n")

            f.write(
                "*Note: Metric MDS preserves the actual distances between points, while non-metric MDS "
                "preserves only the ordering of the distances.*\n\n"
            )

            # Comparison Results
            f.write("## 3. Comparison Results\n\n")

            # Summary statistics
            f.write("### 3.1 Performance Metrics\n\n")

            # Create comparison table
            f.write("| Metric | t-SNE | MDS |\n")
            f.write("|--------|-------|-----|\n")

            # Execution time
            tsne_time = comparison["t-SNE"]["execution_time_seconds"]
            mds_time = comparison["MDS"]["execution_time_seconds"]
            f.write(f"| Execution Time (s) | {tsne_time:.2f} | {mds_time:.2f} |\n")

            # Sample count
            tsne_samples = comparison["t-SNE"]["n_samples"]
            mds_samples = comparison["MDS"]["n_samples"]
            f.write(f"| Number of Samples | {tsne_samples} | {mds_samples} |\n")

            # Cluster separation (silhouette score)
            if (
                comparison["t-SNE"]["cluster_separation"] is not None
                and comparison["MDS"]["cluster_separation"] is not None
            ):
                tsne_silhouette = comparison["t-SNE"]["cluster_separation"]
                mds_silhouette = comparison["MDS"]["cluster_separation"]
                f.write(
                    f"| Silhouette Score | {tsne_silhouette:.3f} | {mds_silhouette:.3f} |\n"
                )

            # Stress (only for MDS)
            if "stress" in comparison["MDS"]:
                mds_stress = comparison["MDS"]["stress"]
                f.write(f"| Stress Value | N/A | {mds_stress:.4f} |\n")

            f.write("\n")

            # Cluster Preservation Analysis
            f.write("### 3.2 Cluster Preservation\n\n")

            if (
                comparison["t-SNE"]["cluster_separation"] is not None
                and comparison["MDS"]["cluster_separation"] is not None
            ):
                tsne_silhouette = comparison["t-SNE"]["cluster_separation"]
                mds_silhouette = comparison["MDS"]["cluster_separation"]

                # Determine which method preserved clusters better
                if tsne_silhouette > mds_silhouette:
                    better_method = "t-SNE"
                    score_diff = tsne_silhouette - mds_silhouette
                else:
                    better_method = "MDS"
                    score_diff = mds_silhouette - tsne_silhouette

                f.write(
                    f"Based on the silhouette scores, **{better_method}** preserved the cluster structure better "
                    f"by a margin of {score_diff:.3f}. "
                )

                # Interpret the scores
                if max(tsne_silhouette, mds_silhouette) > 0.7:
                    f.write(
                        "The highest silhouette score indicates excellent cluster preservation, "
                        "with well-separated and cohesive clusters in the reduced space.\n\n"
                    )
                elif max(tsne_silhouette, mds_silhouette) > 0.5:
                    f.write(
                        "The highest silhouette score indicates good cluster preservation, "
                        "with reasonably separated clusters in the reduced space.\n\n"
                    )
                elif max(tsne_silhouette, mds_silhouette) > 0.3:
                    f.write(
                        "The highest silhouette score indicates fair cluster preservation, "
                        "with moderately separated clusters in the reduced space.\n\n"
                    )
                else:
                    f.write(
                        "Both methods show relatively poor cluster preservation, "
                        "suggesting that the clusters are not well-separated in the reduced space "
                        "or that the original clustering may not be strongly supported by the data.\n\n"
                    )
            else:
                f.write(
                    "Cluster separation metrics are not available for comparison.\n\n"
                )

            # Count countries in each cluster for both methods
            if "cluster" in tsne_df.columns and "cluster" in mds_df.columns:
                f.write("#### Cluster Distribution Comparison\n\n")

                # t-SNE cluster distribution
                tsne_clusters = tsne_df["cluster"].value_counts().sort_index()

                # MDS cluster distribution
                mds_clusters = mds_df["cluster"].value_counts().sort_index()

                # Combine into one table
                f.write("| Cluster | t-SNE Count | t-SNE % | MDS Count | MDS % |\n")
                f.write("|---------|------------|---------|-----------|-------|\n")

                # Get all unique cluster IDs
                all_clusters = sorted(
                    set(tsne_clusters.index) | set(mds_clusters.index)
                )

                for cluster in all_clusters:
                    tsne_count = tsne_clusters.get(cluster, 0)
                    mds_count = mds_clusters.get(cluster, 0)
                    tsne_pct = 100 * tsne_count / len(tsne_df)
                    mds_pct = 100 * mds_count / len(mds_df)
                    f.write(
                        f"| {int(cluster)} | {tsne_count} | {tsne_pct:.1f}% | {mds_count} | {mds_pct:.1f}% |\n"
                    )

                f.write("\n")

            # Interpretation section
            f.write("## 4. Interpretation of Results\n\n")

            # t-SNE assessment
            f.write("### 4.1 t-SNE Performance\n\n")
            f.write(
                "t-SNE is particularly effective at preserving local structure in the data, focusing on keeping similar points close "
                "together in the reduced space. This makes it excellent for visualizing clusters and revealing local patterns.\n\n"
            )

            if (
                comparison["t-SNE"]["execution_time_seconds"]
                < comparison["MDS"]["execution_time_seconds"]
            ):
                f.write(
                    "In this analysis, t-SNE performed faster than MDS, which is unusual for larger datasets. "
                    "This efficiency could be due to the specific implementation or the nature of the data being analyzed.\n\n"
                )
            else:
                f.write(
                    "As expected, t-SNE took longer to execute than MDS, which is typical due to its more complex "
                    "optimization process, especially for larger datasets.\n\n"
                )

            # MDS assessment
            f.write("### 4.2 MDS Performance\n\n")

            metric_type = (
                "metric"
                if comparison["MDS"]["parameters"].get("metric", True)
                else "non-metric"
            )

            f.write(
                f"The {metric_type} MDS implementation used in this analysis focuses on preserving "
                f"{'the actual distances' if metric_type == 'metric' else 'the ordering of distances'} "
                "between points in the high-dimensional space. This approach differs from t-SNE's focus on preserving "
                "local structure.\n\n"
            )

            if "stress" in comparison["MDS"]:
                stress = comparison["MDS"]["stress"]
                if stress < 0.05:
                    stress_quality = "excellent"
                elif stress < 0.1:
                    stress_quality = "good"
                elif stress < 0.2:
                    stress_quality = "fair"
                else:
                    stress_quality = "poor"

                f.write(
                    f"The MDS stress value of {stress:.4f} indicates {stress_quality} fit quality. "
                    f"{'This suggests that the 2D representation accurately preserves the high-dimensional distances.' if stress_quality in ['excellent', 'good'] else 'This suggests some distortion in the 2D representation compared to the high-dimensional distances.'}\n\n"
                )

            # Method Comparison
            f.write("### 4.3 Method Comparison\n\n")

            f.write(
                "Both t-SNE and MDS have their strengths and limitations when visualizing high-dimensional data:\n\n"
                "**t-SNE Advantages:**\n"
                "- Excels at preserving local structure and revealing clusters\n"
                "- Often produces more visually appealing visualizations for exploration\n"
                "- Better at handling non-linear relationships in the data\n\n"
                "**t-SNE Limitations:**\n"
                "- Can't reliably preserve global structure (distances between well-separated clusters)\n"
                "- Results can vary with different random initializations\n"
                "- Parameter sensitive (especially to perplexity)\n"
                "- Typically slower for large datasets\n\n"
                "**MDS Advantages:**\n"
                "- Better at preserving global structure and actual distances\n"
                "- Provides a stress measure to evaluate the quality of the embedding\n"
                "- Results are more stable across runs\n"
                "- Often faster than t-SNE, especially for larger datasets\n\n"
                "**MDS Limitations:**\n"
                "- May not reveal local structure as effectively as t-SNE\n"
                "- Metric MDS assumes distances in the high-dimensional space are meaningful\n"
                "- Can perform poorly when the data has a complex non-linear structure\n\n"
            )

            # Recommendation
            f.write("## 5. Recommendation\n\n")

            # Determine which method to recommend based on the analysis
            if (
                comparison["t-SNE"]["cluster_separation"] is not None
                and comparison["MDS"]["cluster_separation"] is not None
            ):
                tsne_silhouette = comparison["t-SNE"]["cluster_separation"]
                mds_silhouette = comparison["MDS"]["cluster_separation"]

                silhouette_diff = abs(tsne_silhouette - mds_silhouette)

                if silhouette_diff < 0.05:  # Very small difference
                    f.write(
                        "Based on the comparison results, **both methods perform similarly** in terms of cluster preservation. "
                    )

                    # If performance is similar, consider execution time
                    if (
                        comparison["t-SNE"]["execution_time_seconds"]
                        < comparison["MDS"]["execution_time_seconds"]
                    ):
                        f.write(
                            "Since t-SNE executes faster in this case, it might be preferred for efficiency reasons. "
                            "However, MDS provides the additional stress metric to evaluate the quality of the embedding.\n\n"
                        )
                    else:
                        f.write(
                            "Since MDS executes faster, it might be preferred for efficiency reasons. "
                            "Additionally, MDS provides the stress metric to evaluate the quality of the embedding.\n\n"
                        )

                    f.write(
                        "**Recommendation**: Consider using both methods in your analysis to gain complementary insights "
                        "into the data structure. t-SNE might reveal local patterns that MDS misses, while MDS might "
                        "better preserve the global structure.\n\n"
                    )
                else:
                    # Clear winner based on silhouette score
                    better_method = (
                        "t-SNE" if tsne_silhouette > mds_silhouette else "MDS"
                    )

                    if better_method == "t-SNE":
                        f.write(
                            "Based on the comparison results, **t-SNE is recommended** as it shows better cluster preservation "
                            "for this dataset. The higher silhouette score indicates that t-SNE more effectively captures "
                            "the cluster structure in the reduced 2D space.\n\n"
                        )

                        if (
                            comparison["t-SNE"]["execution_time_seconds"]
                            <= comparison["MDS"]["execution_time_seconds"]
                        ):
                            f.write(
                                "t-SNE also executes faster, making it doubly advantageous for this dataset.\n\n"
                            )
                        else:
                            f.write(
                                "While t-SNE takes longer to execute, the improved cluster preservation likely justifies "
                                "the additional computational cost.\n\n"
                            )
                    else:  # MDS is better
                        f.write(
                            "Based on the comparison results, **MDS is recommended** as it shows better cluster preservation "
                            "for this dataset. The higher silhouette score indicates that MDS more effectively captures "
                            "the cluster structure in the reduced 2D space.\n\n"
                        )

                        if (
                            comparison["MDS"]["execution_time_seconds"]
                            <= comparison["t-SNE"]["execution_time_seconds"]
                        ):
                            f.write(
                                "MDS also executes faster, making it doubly advantageous for this dataset.\n\n"
                            )
                        else:
                            f.write(
                                "While MDS takes longer to execute, the improved cluster preservation likely justifies "
                                "the additional computational cost.\n\n"
                            )

                        if (
                            "stress" in comparison["MDS"]
                            and comparison["MDS"]["stress"] < 0.1
                        ):
                            f.write(
                                "Additionally, the low stress value suggests that MDS is effectively preserving "
                                "the high-dimensional distances in the 2D representation.\n\n"
                            )
            else:
                # No silhouette scores available, use execution time and other factors
                if (
                    comparison["t-SNE"]["execution_time_seconds"]
                    < comparison["MDS"]["execution_time_seconds"]
                ):
                    f.write(
                        "Without cluster separation metrics, it's difficult to definitively recommend one method over the other. "
                        "However, t-SNE executes faster for this dataset and typically excels at revealing local structure, "
                        "which might be beneficial for exploratory analysis.\n\n"
                    )
                else:
                    f.write(
                        "Without cluster separation metrics, it's difficult to definitively recommend one method over the other. "
                        "However, MDS executes faster for this dataset and provides a stress metric to evaluate the quality "
                        "of the embedding, which might be beneficial for quantitative analysis.\n\n"
                    )

                f.write(
                    "**Recommendation**: Consider using both methods and examining the visualizations to determine "
                    "which provides more meaningful insights for your specific analysis goals.\n\n"
                )

            # Conclusion
            f.write("## 6. Conclusion\n\n")
            f.write(
                "This comparative analysis of t-SNE and MDS dimensionality reduction techniques provides insights into "
                "their respective strengths, limitations, and performance characteristics when applied to country data.\n\n"
                "The choice between these techniques depends on the specific goals of the analysis:\n\n"
                "- **For cluster visualization and local structure exploration**: t-SNE typically excels at revealing clusters and local patterns.\n"
                "- **For preserving global structure and actual distances**: MDS is often more reliable.\n"
                "- **For computational efficiency**: The faster method depends on the specific dataset and implementation.\n\n"
                "For comprehensive analysis, using both techniques in conjunction can provide complementary perspectives "
                "on the high-dimensional data structure.\n\n"
            )

            f.write("*End of Report*\n")

        print(f"Dimensionality reduction comparison report generated: {report_path}")
        return True

    except Exception as e:
        print(f"Error generating comparison report: {str(e)}")
        import traceback

        traceback.print_exc()
        return False
