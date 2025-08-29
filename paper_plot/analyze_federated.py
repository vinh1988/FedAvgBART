import os
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set up paths
BASE_DIR = Path("/mnt/sda1/Projects/jsl/vp_gitlab/FED/FED-OPT-BERT/commit/FED-OPT-BERT-PYTORCH/paper_plot")
FED_DIR = BASE_DIR / "data_to_plot/generation/federated"
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

def load_federated_results():
    """Load and process federated training results."""
    results = []
    
    # Define the base directory for federated results
    base_fed_dir = FED_DIR / "result_2_to_5_clients"
    
    # Find all model result directories
    for model_dir in base_fed_dir.glob("results_*"):
        model_name = "BART-large" if "bart_large" in model_dir.name.lower() else "DistilBART"
        
        # Process each number of clients
        for nc_dir in model_dir.glob("nc_*"):
            try:
                num_clients = int(nc_dir.name.split('_')[1])
                
                # Process each run
                for run_dir in nc_dir.glob("run_*"):
                    metrics_file = run_dir / "metrics.csv"
                    config_file = run_dir / "config.yaml"
                    
                    if not metrics_file.exists() or not config_file.exists():
                        continue
                        
                    # Load metrics
                    print(f"Processing: {run_dir}")
                    try:
                        df = pd.read_csv(metrics_file)
                        print(f"  - Loaded metrics.csv with {len(df)} rows")
                        
                        # Load config to get alpha value
                        with open(config_file, 'r') as f:
                            config = yaml.safe_load(f)
                            alpha = config.get('dirichlet_alpha', 0.5)  # Default to 0.5 if not found
                            print(f"  - Found alpha: {alpha} in config")
                        
                        # Add metadata
                        df['model'] = model_name
                        df['num_clients'] = num_clients
                        df['alpha'] = alpha
                        print(f"  - Added metadata: model={model_name}, num_clients={num_clients}, alpha={alpha}")
                    except Exception as e:
                        print(f"  - Error processing {run_dir}: {str(e)}")
                        continue
                    
                    # Load client contributions if available
                    contrib_file = run_dir / "client_contributions" / "client_metrics.csv"
                    if contrib_file.exists():
                        contrib_df = pd.read_csv(contrib_file)
                        df['mean_contribution'] = contrib_df['contribution'].mean()
                        df['std_contribution'] = contrib_df['contribution'].std()
                    
                    results.append(df)
                    
            except Exception as e:
                print(f"Error processing {nc_dir}: {e}")
    
    if not results:
        raise ValueError("No federated results found!")
    
    return pd.concat(results, ignore_index=True)

def save_federated_metrics(df, metrics, title_suffix):
    """Save federated training metrics to CSV files."""
    # Save the full metrics data
    output_file = OUTPUT_DIR / f'federated_metrics_{title_suffix}.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved federated metrics to {output_file}")
    
    # Calculate and save mean metrics across runs
    mean_metrics = df.groupby(['model', 'alpha', 'round']).mean(numeric_only=True).reset_index()
    mean_output_file = OUTPUT_DIR / f'federated_mean_metrics_{title_suffix}.csv'
    mean_metrics.to_csv(mean_output_file, index=False)
    print(f"Saved mean federated metrics to {mean_output_file}")
    
    return mean_metrics

def save_client_contributions(df):
    """Save client contributions data to CSV."""
    # Calculate contribution statistics
    contributions = df.groupby(['model', 'alpha']).agg({
        'mean_contribution': ['mean', 'std', 'min', 'max'],
        'gini_coefficient': 'mean',
        'cv_contribution': 'mean'
    }).reset_index()
    
    # Flatten multi-index columns
    contributions.columns = ['_'.join(col).strip('_') for col in contributions.columns.values]
    
    # Save to CSV
    output_file = OUTPUT_DIR / 'client_contributions.csv'
    contributions.to_csv(output_file, index=False)
    print(f"Saved client contributions to {output_file}")
    
    return contributions

def generate_federated_summary(df):
    """Generate summary statistics for federated results."""
    # Get best round for each model and alpha based on rougeL
    best_rounds = df.loc[df.groupby(['model', 'alpha'])['rougeL'].idxmax()]
    
    # Select metrics to include (using bleu4 as the BLEU metric)
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu4', 'train_loss']
    
    # Only include columns that exist in the dataframe
    available_metrics = [m for m in metrics if m in best_rounds.columns]
    summary_cols = ['model', 'alpha', 'round'] + available_metrics
    
    # Create summary with available columns
    summary = best_rounds[summary_cols].copy()
    
    # Rename columns for better display
    column_rename = {
        'model': 'Model',
        'alpha': 'α',
        'round': 'Best Round',
        'train_loss': 'Training Loss'
    }
    
    # Only include columns that exist in the dataframe
    column_rename = {k: v for k, v in column_rename.items() if k in summary.columns}
    summary = summary.rename(columns=column_rename)
    
    # Save to CSV
    output_file = OUTPUT_DIR / 'federated_summary.csv'
    summary.to_csv(output_file, index=False)
    print(f"Saved federated summary to {output_file}")
    
    return summary

def main():
    print("Collecting federated training results...")
    
    # Load and process data
    try:
        df = load_federated_results()
        
        # Save metrics data
        metrics_sets = [
            (['rouge1', 'rouge2', 'rougeL'], 'rouge'),
            (['bleu4'], 'bleu'),  # Using bleu4 as the main BLEU score
            (['train_loss'], 'training')
        ]
        
        # Save metrics for each set
        for metrics, suffix in metrics_sets:
            save_federated_metrics(df, metrics, suffix)
        
        # Save client contributions data
        save_client_contributions(df)
        
        # Generate and save summary
        summary = generate_federated_summary(df)
        
        print("\nSummary of best performance for each model and α:")
        print(summary.to_string(index=False))
        
    except Exception as e:
        print(f"Error during data collection: {e}")
        raise
    
    print(f"\nData collection complete. Results saved to: {OUTPUT_DIR}")
    return df

if __name__ == "__main__":
    main()
