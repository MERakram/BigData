import pandas as pd
import traceback
from datetime import datetime


def generate_markdown_report(df, results, availability_stats):
    """Generate a comprehensive markdown report of the analysis

    Args:
        df (pandas.DataFrame): DataFrame containing the analyzed country data
        results (dict): Dictionary containing analysis results (basic_stats, correlations, clusters, etc.)
        availability_stats (dict): Dictionary containing data availability statistics

    Returns:
        bool: True if report generation was successful, False otherwise
    """
    report_path = "../../data/output/country_analysis_report.md"

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            # Report header
            f.write(
                "# Country Analysis Report: Happiness, GDP, and Currency Strength\n\n"
            )
            f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")

            # Introduction
            f.write("## 1. Introduction\n\n")
            f.write(
                "This report analyzes the relationships between happiness scores, GDP per capita, and currency strength "
            )
            f.write(
                "across different countries. The analysis aims to answer the research question:\n\n"
            )
            f.write(
                '> "How do happiness, GDP, and currency strength relate to each other?"\n\n'
            )

            # Data overview
            f.write("## 2. Data Overview\n\n")
            f.write(
                f"The analysis includes data from **{len(df)}** countries worldwide. "
            )
            f.write("The dataset contains the following key metrics:\n\n")
            f.write(
                "- **Happiness Score**: World Happiness Report scores (scale of 0-10)\n"
            )
            f.write(
                "- **GDP per Capita (EUR)**: Gross Domestic Product per person in Euros\n"
            )
            f.write(
                "- **Exchange Rate to EUR**: Currency strength relative to the Euro\n\n"
            )

            f.write("### 2.1 Data Availability\n\n")
            f.write("| Metric | Available Countries | Percentage |\n")
            f.write("|--------|---------------------|------------|\n")
            for metric, stats in availability_stats.items():
                f.write(
                    f"| {metric} | {stats['count']}/{len(df)} | {stats['percentage']:.1f}% |\n"
                )

            # Summary statistics
            f.write("\n## 3. Summary Statistics\n\n")
            f.write("### 3.1 Basic Statistics\n\n")

            stats_df = pd.DataFrame(results["basic_stats"])
            f.write(f"```\n{stats_df.to_string()}\n```\n\n")

            # Data interpretation
            f.write("### 3.2 Key Observations\n\n")

            # Get happiness stats
            happiness_mean = stats_df.loc["mean", "happiness_score"]
            happiness_min = stats_df.loc["min", "happiness_score"]
            happiness_max = stats_df.loc["max", "happiness_score"]
            gdp_mean = stats_df.loc["mean", "gdp_per_capita_eur"]
            gdp_max = stats_df.loc["max", "gdp_per_capita_eur"]

            f.write(
                f"- The average happiness score across countries is **{happiness_mean:.2f}** (on a scale of 0-10).\n"
            )
            f.write(
                f"- Happiness scores range from **{happiness_min:.2f}** to **{happiness_max:.2f}**.\n"
            )
            f.write(f"- The average GDP per capita is **€{gdp_mean:,.2f}**.\n")
            f.write(f"- The highest GDP per capita is **€{gdp_max:,.2f}**.\n\n")

            # Correlations
            f.write("## 4. Correlation Analysis\n\n")
            corr_matrix = pd.DataFrame(results["correlations"])
            f.write("### 4.1 Correlation Matrix\n\n")
            f.write(f"```\n{corr_matrix.to_string()}\n```\n\n")

            # Write correlation interpretations
            f.write("### 4.2 Interpretation\n\n")

            happiness_gdp_corr = corr_matrix.loc[
                "happiness_score", "gdp_per_capita_eur"
            ]
            happiness_exchange_corr = corr_matrix.loc[
                "happiness_score", "exchange_rate_eur"
            ]
            gdp_exchange_corr = corr_matrix.loc[
                "gdp_per_capita_eur", "exchange_rate_eur"
            ]

            f.write(
                f"- **Happiness and GDP per Capita**: Correlation coefficient = **{happiness_gdp_corr:.3f}**\n"
            )
            if happiness_gdp_corr > 0.7:
                f.write(
                    "  - There is a strong positive correlation between happiness and GDP per capita.\n"
                )
                f.write(
                    "  - Countries with higher GDP per capita tend to report higher happiness scores.\n"
                )
            elif happiness_gdp_corr > 0.3:
                f.write(
                    "  - There is a moderate positive correlation between happiness and GDP per capita.\n"
                )
                f.write(
                    "  - Higher GDP per capita is somewhat associated with higher happiness levels.\n"
                )
            elif happiness_gdp_corr > 0:
                f.write(
                    "  - There is a weak positive correlation between happiness and GDP per capita.\n"
                )
                f.write(
                    "  - GDP per capita has a slight positive association with happiness scores.\n"
                )
            else:
                f.write(
                    "  - There is no positive correlation between happiness and GDP per capita.\n"
                )
                f.write(
                    "  - Higher GDP does not necessarily indicate higher happiness scores in this dataset.\n"
                )

            f.write(
                f"\n- **Happiness and Exchange Rate**: Correlation coefficient = **{happiness_exchange_corr:.3f}**\n"
            )
            if abs(happiness_exchange_corr) > 0.7:
                strength = "strong"
            elif abs(happiness_exchange_corr) > 0.3:
                strength = "moderate"
            else:
                strength = "weak"

            direction = "positive" if happiness_exchange_corr > 0 else "negative"
            f.write(
                f"  - There is a {strength} {direction} correlation between happiness and currency exchange rate.\n"
            )
            if happiness_exchange_corr > 0:
                f.write(
                    "  - Countries with stronger currencies (higher EUR exchange rates) tend to have higher happiness scores.\n"
                )
            else:
                f.write(
                    "  - Currency strength alone does not show a clear positive relationship with happiness.\n"
                )

            f.write(
                f"\n- **GDP per Capita and Exchange Rate**: Correlation coefficient = **{gdp_exchange_corr:.3f}**\n"
            )
            if abs(gdp_exchange_corr) > 0.7:
                f.write(
                    "  - There is a strong relationship between GDP per capita and currency strength.\n"
                )
                if gdp_exchange_corr > 0:
                    f.write(
                        "  - Countries with stronger currencies tend to have higher GDP per capita.\n"
                    )
                else:
                    f.write(
                        "  - Interestingly, higher GDP per capita countries don't necessarily have stronger currencies relative to EUR.\n"
                    )
            elif abs(gdp_exchange_corr) > 0.3:
                f.write(
                    "  - There is a moderate relationship between GDP per capita and currency strength.\n"
                )
            else:
                f.write(
                    "  - There is only a weak relationship between GDP per capita and currency strength.\n"
                )

            # Cluster analysis
            if results["clusters"] is not None:
                f.write("\n## 5. Cluster Analysis\n\n")
                f.write(
                    "K-means clustering was applied to identify natural groupings in the data based on happiness, GDP per capita, and exchange rates.\n\n"
                )

                # Describe clusters
                f.write("### 5.1 Cluster Characteristics\n\n")
                f.write(
                    "| Cluster | Count | Avg Happiness | Avg GDP per Capita (EUR) | Avg Exchange Rate |\n"
                )
                f.write(
                    "|---------|-------|---------------|--------------------------|------------------|\n"
                )

                # Group data by cluster
                if "cluster" in df.columns:
                    cluster_summary = df.groupby("cluster")[
                        ["happiness_score", "gdp_per_capita_eur", "exchange_rate_eur"]
                    ].mean()
                    cluster_counts = df["cluster"].value_counts().sort_index()

                    for cluster_id in sorted(cluster_summary.index):
                        if pd.notna(
                            cluster_id
                        ):  # Only include rows where cluster is not NaN
                            row = cluster_summary.loc[int(cluster_id)]
                            count = cluster_counts.get(cluster_id, 0)
                            f.write(
                                f"| {int(cluster_id)} | {count} | {row['happiness_score']:.2f} | {row['gdp_per_capita_eur']:,.2f} | {row['exchange_rate_eur']:.4f} |\n"
                            )

                # Describe what each cluster represents
                f.write("\n### 5.2 Cluster Interpretation\n\n")

                # This requires some analysis to generate meaningful descriptions
                if "cluster" in df.columns:
                    clusters = df.groupby("cluster")

                    for cluster_id in sorted(clusters.groups.keys()):
                        if pd.notna(cluster_id):
                            cluster_df = clusters.get_group(cluster_id)

                            # Calculate metrics
                            avg_happiness = cluster_df["happiness_score"].mean()
                            avg_gdp = cluster_df["gdp_per_capita_eur"].mean()
                            avg_exchange = cluster_df["exchange_rate_eur"].mean()

                            # Get sample countries (top 5 by population)
                            sample_countries = (
                                cluster_df.sort_values("population", ascending=False)
                                .head(5)["country"]
                                .tolist()
                            )

                            # Determine cluster characteristics
                            if (
                                avg_happiness
                                > results["basic_stats"]["happiness_score"]["mean"]
                            ):
                                happiness_level = "high"
                            else:
                                happiness_level = "lower"

                            if (
                                avg_gdp
                                > results["basic_stats"]["gdp_per_capita_eur"]["mean"]
                            ):
                                wealth_level = "wealthy"
                            else:
                                wealth_level = "less wealthy"

                            f.write(
                                f"**Cluster {int(cluster_id)}**: This group represents countries with {happiness_level} happiness scores and {wealth_level} economies.\n"
                            )
                            f.write(
                                f"- **Notable countries**: {', '.join(sample_countries[:3])}\n"
                            )
                            f.write(
                                f"- **Average happiness score**: {avg_happiness:.2f}\n"
                            )
                            f.write(f"- **Average GDP per capita**: €{avg_gdp:,.2f}\n")
                            f.write(
                                f"- **Average exchange rate**: {avg_exchange:.4f}\n\n"
                            )

                # Add visualization
                f.write("### 5.3 Visualization\n\n")
                f.write("![Happiness vs GDP Clusters](happiness_gdp_clusters.png)\n\n")
                f.write(
                    "*Figure: Scatter plot showing happiness scores vs GDP per capita, colored by cluster assignment.*\n\n"
                )

            # Conclusions
            f.write("## 6. Conclusions\n\n")

            # Generate conclusions based on correlations
            if happiness_gdp_corr > 0.5:
                f.write(
                    "1. **Economic prosperity correlates with happiness**: The data shows a significant positive correlation between GDP per capita and happiness scores, suggesting that economic well-being is an important factor in overall life satisfaction.\n\n"
                )
            else:
                f.write(
                    "1. **Economic prosperity shows some correlation with happiness**: While there is a correlation between GDP per capita and happiness, it's not as strong as might be expected, suggesting that factors beyond wealth contribute significantly to happiness.\n\n"
                )

            if happiness_exchange_corr > 0.3:
                f.write(
                    "2. **Currency strength appears relevant to happiness**: Countries with stronger currencies relative to the Euro tend to report higher happiness levels, which may reflect broader economic stability.\n\n"
                )
            else:
                f.write(
                    "2. **Currency strength shows limited correlation with happiness**: The relationship between exchange rates and happiness scores is not strong, indicating that currency strength alone is not a primary driver of happiness.\n\n"
                )

            # Additional general conclusions
            f.write(
                "3. **Multiple factors influence happiness**: The clustering analysis reveals distinct groups of countries with different combinations of economic prosperity and happiness levels, showing that the relationship between wealth and happiness is complex and multifaceted.\n\n"
            )

            f.write(
                "4. **Future research directions**: Further investigation could explore additional factors like income inequality, social support systems, healthcare access, and political freedom, which might explain variations in happiness that aren't accounted for by economic indicators alone.\n\n"
            )

            # Methodology note
            f.write("## 7. Methodology\n\n")
            f.write("This analysis was conducted using the following steps:\n\n")
            f.write(
                "1. Data was loaded from normalized country datasets containing happiness scores, economic indicators, and currency exchange rates.\n"
            )
            f.write("2. Relevant metrics were extracted and cleaned for analysis.\n")
            f.write(
                "3. Descriptive statistics and correlations were calculated to identify relationships between variables.\n"
            )
            f.write(
                "4. K-means clustering was applied to identify natural groupings in the data.\n"
            )
            f.write(
                "5. Results were visualized and interpreted to draw conclusions about the relationships between happiness, GDP, and currency strength.\n\n"
            )

            f.write("*End of Report*\n")

        print(f"\nMarkdown report generated successfully: {report_path}")
        return True

    except Exception as e:
        print(f"Error generating markdown report: {str(e)}")
        traceback.print_exc()
        return False
