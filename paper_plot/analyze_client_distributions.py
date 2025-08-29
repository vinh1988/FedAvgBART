#!/usr/bin/env python3
"""
Script to analyze and visualize the data distribution across clients in federated learning.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set up paths
base_dir = Path('/mnt/sda1/Projects/jsl/vp_gitlab/FED/FED-OPT-BERT/commit/FED-OPT-BERT-PYTORCH/paper_plot')
output_dir = base_dir / 'analysis_results/classification/federated/client_distributions'
output_dir.mkdir(parents=True, exist_ok=True)

# Find all data distribution files
dist_files = list(Path(base_dir).glob('**/data_distribution_train.csv'))
print(f"Found {len(dist_files)} data distribution files")

# Initialize data storage
data = []

# Process each distribution file
for file_path in dist_files:
    # Extract metadata from path
    path_parts = file_path.parts
    try:
        model = 'BART-large' if 'bart_large' in str(file_path).lower() else 'DistilBART'
        num_clients = int(path_parts[-3].split('_')[1])  # clients_X_rounds_22 -> X
        run_id = path_parts[-2]  # The timestamp directory
        
        # Read the distribution file
        df = pd.read_csv(file_path)
        
        # Get alpha value (should be the same for all clients in a run)
        alpha = df['dirichlet_alpha'].iloc[0] if 'dirichlet_alpha' in df.columns else 0.1
        
        # Add to our data collection
        for _, row in df.iterrows():
            client_data = {
                'model': model,
                'num_clients': num_clients,
                'run_id': run_id,
                'alpha': alpha,
                'client_id': row['client_id'],
                'num_samples': row['num_samples'],
                'kl_to_global': row['kl_to_global']
            }
            
            # Add class distribution
            for i in range(20):  # Assuming 20 classes (class_0 to class_19)
                class_col = f'class_{i}'
                if class_col in row:
                    client_data[f'class_{i}'] = row[class_col]
                    
            data.append(client_data)
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Convert to DataFrame
if not data:
    print("No data found!")
    exit()

df = pd.DataFrame(data)

# Save the combined data
combined_file = output_dir / 'client_distributions_combined.csv'
df.to_csv(combined_file, index=False)
print(f"Saved combined client distributions to {combined_file}")

# Generate summary statistics
summary = df.groupby(['model', 'num_clients', 'alpha']).agg(
    total_clients=('client_id', 'count'),
    avg_samples=('num_samples', 'mean'),
    avg_kl=('kl_to_global', 'mean')
).reset_index()

# Save summary
summary_file = output_dir / 'client_distributions_summary.csv'
summary.to_csv(summary_file, index=False)
print(f"Saved client distributions summary to {summary_file}")

# Create visualizations
def plot_class_distribution(data, title, filename):
    """Plot class distribution across clients."""
    plt.figure(figsize=(15, 8))
    
    # Get class columns
    class_cols = [f'class_{i}' for i in range(20) if f'class_{i}' in data.columns]
    
    # Calculate mean and std of class distribution
    class_means = data[class_cols].mean()
    class_stds = data[class_cols].std()
    
    # Plot
    x = np.arange(len(class_cols))
    plt.bar(x, class_means, yerr=class_stds, capsize=5, alpha=0.7)
    
    plt.xticks(x, [f'Class {i}' for i in range(20)], rotation=45, ha='right')
    plt.ylabel('Average number of samples per client')
    plt.title(title)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

# Plot overall class distribution
plot_class_distribution(df, 'Overall Class Distribution Across All Clients', 'overall_class_distribution.png')

# Plot class distribution by alpha
for alpha in df['alpha'].unique():
    alpha_data = df[df['alpha'] == alpha]
    plot_class_distribution(
        alpha_data, 
        f'Class Distribution (Alpha={alpha})', 
        f'class_distribution_alpha_{alpha}.png'
    )

# Plot KL divergence by number of clients and alpha
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='num_clients', y='kl_to_global', hue='alpha')
plt.title('KL Divergence from Global Distribution by Number of Clients and Alpha')
plt.xlabel('Number of Clients')
plt.ylabel('KL Divergence from Global')
plt.legend(title='Alpha')
plt.tight_layout()
plt.savefig(output_dir / 'kl_divergence_by_clients_alpha.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"All visualizations saved to {output_dir}")

# ==============================================================
# 100%-stacked bars: per-client share for each N clients
# One bar per N (e.g., 2..10), stacked segments for each client i in [1..N]
# ==============================================================

def plot_client_share_stacked(df_in: pd.DataFrame, n_min: int, n_max: int, filename: str, title: str):
    """Create a 100%-stacked bar chart: one bar per N in [n_min..n_max],
    where each bar is split into N segments showing each client's percentage share.

    Note: If multiple runs exist for the same N, we average shares per client index.
    """
    # Work on a copy
    d = df_in.copy()
    # Ensure expected columns
    required_cols = {"num_clients", "client_id", "num_samples"}
    if not required_cols.issubset(d.columns):
        print(f"Skipping stacked share plot; missing columns in DataFrame: {required_cols - set(d.columns)}")
        return

    # Normalize client_id per N to be 1..N (for labeling). If client_id starts at 0, add 1 for display
    # We'll keep original client_id for grouping, but create a display label
    d["client_label"] = d["client_id"].astype(int) + 1

    # Compute share per run and N: share = num_samples / total_samples_in_run_for_that_N
    group_cols = ["num_clients", "run_id", "client_id", "client_label"] if "run_id" in d.columns else ["num_clients", "client_id", "client_label"]
    sum_cols = ["num_clients", "run_id"] if "run_id" in d.columns else ["num_clients"]

    # total per (N, run)
    totals = d.groupby(sum_cols)["num_samples"].sum().reset_index().rename(columns={"num_samples": "total_samples"})
    d = d.merge(totals, on=sum_cols, how="left")
    d["share"] = d["num_samples"] / d["total_samples"].replace(0, pd.NA)

    # Average share per client index for each N across runs
    avg_cols = ["num_clients", "client_id", "client_label"]
    shares = d.groupby(avg_cols)["share"].mean().reset_index()

    # Pivot to wide: rows=N, columns=client_label, values=share
    pivot = shares.pivot(index="num_clients", columns="client_label", values="share").sort_index()

    # Limit N range
    pivot = pivot[(pivot.index >= n_min) & (pivot.index <= n_max)]
    if pivot.empty:
        print(f"No data in requested N range [{n_min}, {n_max}] for stacked share plot.")
        return

    # Plot stacked bars
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot.index))
    # Restrict client columns to those that can appear within the selected N range
    # e.g., for 2..5 keep only client labels 1..5
    max_n = int(pivot.index.max())
    client_cols = [c for c in pivot.columns if int(c) <= max_n]
    pivot = pivot[client_cols]

    # Choose a color palette large enough
    palette = sns.color_palette("tab20", n_colors=max(len(client_cols), 2))

    for i, col in enumerate(client_cols):
        vals = pivot[col].fillna(0.0).values
        plt.bar(x, vals, bottom=bottom, color=palette[i % len(palette)], edgecolor="white", linewidth=0.5, label=f"Client {int(col)}")
        bottom += vals

    plt.xticks(x, [f"{int(n)} clients" for n in pivot.index])
    plt.ylabel("Client share (%)")
    plt.title(title)
    # Convert y-axis to percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y*100)}%"))
    # Legend: keep small to avoid clutter
    if len(client_cols) <= 10:
        plt.legend(ncol=min(len(client_cols), 5), fontsize=8)
    else:
        plt.legend().remove()
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()

# Generate the stacked share plots for ranges 2..10 and 2..5
try:
    plot_client_share_stacked(
        df_in=df,
        n_min=2,
        n_max=10,
        filename="client_share_stacked_2_10.png",
        title="Per-client data share (100%-stacked) across N clients (2..10)",
    )
    plot_client_share_stacked(
        df_in=df,
        n_min=2,
        n_max=5,
        filename="client_share_stacked_2_5.png",
        title="Per-client data share (100%-stacked) across N clients (2..5)",
    )
    print(f"Saved stacked client share plots to {output_dir}")
except Exception as e:
    print(f"Error generating stacked client share plots: {e}")
