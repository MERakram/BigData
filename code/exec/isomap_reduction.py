import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap

def print_feature_availability(df, features):
    print("Feature availability in dataset:")
    for feat in features:
        if feat in df.columns:
            non_null = df[feat].notnull().sum()
            total = len(df)
            pct = non_null / total * 100
            print(f"  {feat}: {non_null}/{total} non-null values ({pct:.1f}%)")
        else:
            print(f"  {feat}: Column not found in the dataset!")
    print()

def main():
    print("Starting Isomap dimensionality reduction analysis...")

    # 1. Load the enriched dataset (adjust file path/format as needed)
    df = pd.read_csv("../../data/input/world-data-2023.csv")
    # Normalize column names: lower-case and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    total = len(df)
    print(f"Loaded dataset with {total} entries")
    print("Available columns:", df.columns.tolist())
    
    # 2. Define the features to use.
    # Adjust these if your CSV file uses different column names.
    features = ["happiness_score", "exchange_rate_eur"]
    
    print_feature_availability(df, features)
    
    # Check if required features exist
    missing_features = [feat for feat in features if feat not in df.columns]
    if missing_features:
        print(f"Error: Missing required columns: {missing_features}")
        return

    # 3. Drop rows with missing values in the selected features
    df_clean = df.dropna(subset=features).copy()
    clean_count = len(df_clean)
    print(f"Complete rows for selected features: {clean_count}/{total}")

    # 4. (Optional) Impute missing values instead of dropping rows
    # imputer = SimpleImputer(strategy='mean')
    # df_clean[features] = imputer.fit_transform(df_clean[features])

    # 5. Extract numeric data and scale it
    imputer = SimpleImputer(strategy='mean')
    data_filled = imputer.fit_transform(df_clean[features])
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_filled)

    # 6. Apply Isomap
    n_neighbors = 5  # Adjust this parameter if needed
    isomap = Isomap(n_components=2, n_neighbors=n_neighbors)
    isomap_result = isomap.fit_transform(data_scaled)
    print(f"Applied Isomap on {clean_count} entries with {len(features)} features.")

    # 7. Save the 2D embedding to a CSV file
    output_dir = "../../data/output"
    os.makedirs(output_dir, exist_ok=True)
    isomap_df = pd.DataFrame(isomap_result, columns=["Isomap1", "Isomap2"])
    # If your dataset contains a 'country' column, add it for reference.
    if "country" in df_clean.columns:
        isomap_df["country"] = df_clean["country"].values
    isomap_csv_path = os.path.join(output_dir, "isomap_results.csv")
    isomap_df.to_csv(isomap_csv_path, index=False)
    print(f"Isomap numerical results saved to {isomap_csv_path}")

    # 8. Create and save the visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(isomap_result[:, 0], isomap_result[:, 1], c='red', alpha=0.6)
    plt.title("Isomap Visualization")
    plt.xlabel("Isomap Component 1")
    plt.ylabel("Isomap Component 2")
    isomap_plot_path = os.path.join(output_dir, "isomap_plot.png")
    plt.savefig(isomap_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Isomap plot saved to {isomap_plot_path}")

    print("Isomap dimensionality reduction analysis complete.")

if __name__ == "__main__":
    main()
