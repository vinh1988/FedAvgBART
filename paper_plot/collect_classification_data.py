#!/usr/bin/env python3
"""
Script to collect and organize classification training data into analysis_results.
"""

import os
import pandas as pd
from pathlib import Path

def process_centralized_data(input_dir, output_dir):
    """Process centralized training metrics."""
    print("Processing centralized training data...")
    centralized_dir = input_dir / 'centralized'
    output_path = output_dir / 'centralized'
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    
    # Process each model's training metrics
    for model_file in centralized_dir.glob('**/training_metrics_*.csv'):
        model_name = 'BART-large' if 'bart_large' in str(model_file).lower() else 'DistilBART'
        
        df = pd.read_csv(model_file)
        df['model'] = model_name
        all_metrics.append(df)
    
    if all_metrics:
        combined_df = pd.concat(all_metrics, ignore_index=True)
        output_file = output_path / 'training_metrics_combined.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"Saved combined centralized metrics to {output_file}")
    
    return combined_df if all_metrics else None

def get_alpha_from_run_dir(run_dir):
    """Extract alpha value from data distribution file in the run directory."""
    # Look for data distribution files
    dist_files = list(run_dir.glob('*data_distribution_train.csv'))
    if not dist_files:
        print(f"Warning: No data distribution file found in {run_dir}")
        return None
    
    try:
        # Read the first line of the first data distribution file
        with open(dist_files[0], 'r') as f:
            header = f.readline().strip()
            if 'dirichlet_alpha' in header:
                # Read the first data line to get alpha
                data_line = f.readline().strip()
                alpha = float(data_line.split(',')[-1])
                return alpha
    except Exception as e:
        print(f"Error extracting alpha from {dist_files[0]}: {e}")
    
    return None

def process_federated_data(input_dir, output_dir):
    """Process federated training metrics."""
    print("Processing federated training data...")
    federated_dir = input_dir / 'federated'
    output_path = output_dir / 'federated'
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    
    # Process each model's federated runs
    for model_dir in federated_dir.glob('result_*'):
        for run_dir in model_dir.glob('results_*'):
            model_name = 'BART-large' if 'bart_large' in str(run_dir).lower() else 'DistilBART'
            print(f"Processing model: {model_name}")

            # Process each client configuration
            for client_dir in run_dir.glob('clients_*'):
                if not client_dir.is_dir():
                    continue

                try:
                    num_clients = int(client_dir.name.split('_')[1])
                except (IndexError, ValueError):
                    print(f"Skipping directory with unexpected name format: {client_dir}")
                    continue

                # Prepare alpha inference map per client_dir when missing
                # Rule (per user): within each clients_X directory, smaller run-folder name -> alpha 0.1,
                # larger run-folder name -> alpha 0.5. Only applied when alpha cannot be read from files.
                run_dirs = [d for d in client_dir.iterdir() if d.is_dir()]
                # Sort by name numerically when possible, else lexicographically
                def sort_key(p):
                    try:
                        # names like YYYYMMDD_HHMMSS
                        return int(p.name.replace('_', ''))
                    except Exception:
                        return p.name
                run_dirs_sorted = sorted(run_dirs, key=sort_key)
                inferred_alpha_by_run = {}
                if len(run_dirs_sorted) >= 2:
                    # Map smallest to 0.1, largest to 0.5; if more than 2, map first half to 0.1, second half to 0.5
                    mid = len(run_dirs_sorted) // 2
                    first_half = run_dirs_sorted[:mid]
                    second_half = run_dirs_sorted[mid:]
                    for r in first_half:
                        inferred_alpha_by_run[r.name] = 0.1
                    for r in second_half:
                        inferred_alpha_by_run[r.name] = 0.5

                # Process each run
                run_count = 0
                for run in run_dirs_sorted:
                    # Get alpha value for this run
                    alpha = get_alpha_from_run_dir(run)
                    if alpha is None:
                        # If model is DistilBART, try folder-name rule; else default to 0.1
                        if model_name == 'DistilBART' and run.name in inferred_alpha_by_run:
                            alpha = inferred_alpha_by_run[run.name]
                            print(f"Info: Inferred alpha={alpha} for {run} by run-folder ordering rule")
                        else:
                            print(f"Warning: Could not determine alpha for {run}, defaulting to 0.1")
                            alpha = 0.1

                    # Look for training metrics files
                    for metrics_file in run.glob('training_metrics_*.csv'):
                        try:
                            df = pd.read_csv(metrics_file)
                            if not df.empty:
                                # Add metadata
                                df['model'] = model_name
                                df['num_clients'] = num_clients
                                df['run_id'] = run.name
                                df['alpha'] = alpha  # Add alpha value

                                # Add phase information if not present
                                if 'phase' not in df.columns:
                                    df['phase'] = 'validation'  # Default to validation if not specified

                                all_metrics.append(df)
                                run_count += 1
                        except Exception as e:
                            print(f"Error processing {metrics_file}: {e}")

                print(f"Processed {run_count} runs for {model_name} with {num_clients} clients")
    
    if not all_metrics:
        print("No federated metrics found")
        return None
    
    # Combine all metrics
    combined_df = pd.concat(all_metrics, ignore_index=True)
    
    # Save combined metrics
    combined_file = output_path / 'training_metrics_combined.csv'
    combined_df.to_csv(combined_file, index=False)
    print(f"Saved combined federated metrics to {combined_file}")
    
    # Generate summary statistics by model, num_clients, and alpha
    summary_stats = combined_df.groupby(['model', 'num_clients', 'alpha'])['accuracy', 'f1', 'precision', 'recall', 'loss'].agg(
        ['mean', 'std', 'count']
    ).round(4)
    
    # Flatten multi-index columns
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats = summary_stats.reset_index()
    
    # Save summary statistics
    summary_file = output_path / 'training_metrics_summary.csv'
    summary_stats.to_csv(summary_file, index=False)
    print(f"Saved federated metrics summary to {summary_file}")
    
    # Find best round for each model, client count, and alpha
    best_rounds = combined_df.loc[combined_df.groupby(['model', 'num_clients', 'alpha', 'run_id'])['accuracy'].idxmax()]
    best_rounds = best_rounds[['model', 'num_clients', 'alpha', 'run_id', 'round', 'accuracy', 'f1', 'precision', 'recall', 'loss']]
    
    # Save best rounds
    best_rounds_file = output_path / 'best_rounds.csv'
    best_rounds.to_csv(best_rounds_file, index=False)
    print(f"Saved best rounds to {best_rounds_file}")
    
    # Generate alpha-specific summaries
    for alpha in combined_df['alpha'].unique():
        alpha_df = combined_df[combined_df['alpha'] == alpha]
        alpha_summary = alpha_df.groupby(['model', 'num_clients'])['accuracy', 'f1', 'precision', 'recall', 'loss'].agg(
            ['mean', 'std', 'count']
        ).round(4)
        alpha_summary.columns = ['_'.join(col).strip() for col in alpha_summary.columns.values]
        alpha_summary = alpha_summary.reset_index()
        alpha_summary_file = output_path / f'training_metrics_alpha_{alpha}.csv'
        alpha_summary.to_csv(alpha_summary_file, index=False)
        print(f"Saved alpha={alpha} summary to {alpha_summary_file}")
    
    return combined_df

def main():
    # Set up paths
    base_dir = Path('/mnt/sda1/Projects/jsl/vp_gitlab/FED/FED-OPT-BERT/commit/FED-OPT-BERT-PYTORCH/paper_plot')
    input_dir = base_dir / 'data_to_plot/classification'
    output_dir = base_dir / 'analysis_results/classification'
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process data
    print("Starting data collection...")
    centralized_df = process_centralized_data(input_dir, output_dir)
    federated_df = process_federated_data(input_dir, output_dir)
    
    print("\nData collection complete!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
