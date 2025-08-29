import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths and directories
BASE_DIR = Path("/mnt/sda1/Projects/jsl/vp_gitlab/FED/FED-OPT-BERT/commit/FED-OPT-BERT-PYTORCH/paper_plot")
DATA_DIR = BASE_DIR / "data_to_plot/generation/central/central"
OUTPUT_DIR = BASE_DIR / "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def load_centralized_data():
    """Load and process centralized training results."""
    # Load data
    bart_df = pd.read_csv(DATA_DIR / "results_bart_large_cnndm_centralized.csv")
    distilbart_df = pd.read_csv(DATA_DIR / "results_distilbart_cnndm_centralized.csv")
    
    # Add model type
    bart_df['model_type'] = 'BART-large'
    distilbart_df['model_type'] = 'DistilBART'
    
    # Combine dataframes
    df = pd.concat([bart_df, distilbart_df])
    
    return df

def save_centralized_metrics(df, title_suffix):
    """Save centralized training metrics to CSV."""
    # Save the full metrics data
    df.to_csv(OUTPUT_DIR / f'centralized_metrics_{title_suffix}.csv', index=False)
    
    # Create a summary of the best performance for each model
    best_metrics = df.loc[df.groupby('model_type')['rougeL'].idxmax()]
    best_metrics.to_csv(OUTPUT_DIR / f'centralized_best_metrics_{title_suffix}.csv', index=False)
    
    return best_metrics

def generate_centralized_summary(df):
    """Generate summary statistics for centralized results."""
    # Get best epoch for each model
    best_epochs = df.loc[df.groupby('model_type')['rougeL'].idxmax()]
    
    # Select metrics to include
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor', 'train_loss', 'val_loss']
    
    # Only include columns that exist in the dataframe
    available_metrics = [m for m in metrics if m in best_epochs.columns]
    summary = best_epochs[['model_type'] + available_metrics].copy()
    
    # Rename columns for better display
    column_rename = {
        'model_type': 'Model',
        'train_loss': 'Training Loss',
        'val_loss': 'Validation Loss'
    }
    
    # Only include columns that exist in the dataframe
    column_rename = {k: v for k, v in column_rename.items() if k in summary.columns}
    summary = summary.rename(columns=column_rename)
    
    # Save to CSV
    summary.to_csv(OUTPUT_DIR / 'centralized_summary.csv', index=False)
    
    return summary

def main():
    print("Analyzing centralized training results...")
    
    # Load and process data
    df = load_centralized_data()
    
    # Save metrics to CSV files
    best_metrics = save_centralized_metrics(df, 'all_metrics')
    
    # Generate summary table
    summary = generate_centralized_summary(df)
    
    print("\nSummary of best performance for each model:")
    print(summary.to_string(index=False))
    
    print(f"\nData collection complete. Results saved to: {OUTPUT_DIR}")
    
    return summary

if __name__ == "__main__":
    main()
