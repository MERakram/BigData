import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, MDS

# 1. Load your enriched dataset (change file name if needed)
df = pd.read_csv("../../data/input/world-data-2023.csv")

# 2. Select only the numerical columns for simplicity
data_numeric = df.select_dtypes(include=['float64', 'int64'])

# 3. Impute missing values (using mean strategy)
imputer = SimpleImputer(strategy='mean')
data_filled = imputer.fit_transform(data_numeric)

# 4. Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_filled)

# 5. Create output directory if it doesn't exist
output_dir = "../../data/output"
os.makedirs(output_dir, exist_ok=True)

############################################
# Method 1: t-SNE
############################################
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
data_tsne = tsne.fit_transform(data_scaled)

# Save t-SNE result as CSV
tsne_df = pd.DataFrame(data_tsne, columns=['TSNE1', 'TSNE2'])
tsne_csv_path = os.path.join(output_dir, "tsne_results.csv")
tsne_df.to_csv(tsne_csv_path, index=False)

# Plot t-SNE results and save as PNG
plt.figure(figsize=(8, 6))
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c='blue', alpha=0.6)
plt.title("t-SNE Visualization")
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
tsne_plot_path = os.path.join(output_dir, "tsne_plot.png")
plt.savefig(tsne_plot_path)
plt.close()

############################################
# Method 2: MDS
############################################
mds = MDS(n_components=2, random_state=42)
data_mds = mds.fit_transform(data_scaled)

# Save MDS result as CSV
mds_df = pd.DataFrame(data_mds, columns=['MDS1', 'MDS2'])
mds_csv_path = os.path.join(output_dir, "mds_results.csv")
mds_df.to_csv(mds_csv_path, index=False)

# Plot MDS results and save as PNG
plt.figure(figsize=(8, 6))
plt.scatter(data_mds[:, 0], data_mds[:, 1], c='green', alpha=0.6)
plt.title("MDS Visualization")
plt.xlabel("MDS1")
plt.ylabel("MDS2")
mds_plot_path = os.path.join(output_dir, "mds_plot.png")
plt.savefig(mds_plot_path)
plt.close()
