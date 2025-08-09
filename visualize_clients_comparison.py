#!/usr/bin/env python3
"""
Visualize comparison of ROUGE metrics across different numbers of clients.
"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(base_dir):
    """Load results from all client experiments."""
    results = []
    
    # Find all metrics files
    metrics_files = list(Path(base_dir).glob('clients_*/fed_distilbart_*_metrics_*.csv'))
    
    for file in metrics_files:
        # Extract number of clients from directory name
        num_clients = int(file.parent.name.split('_')[1])
        
        # Read metrics
        try:
            df = pd.read_csv(file)
            # Get the best ROUGE scores from the last round
            last_round = df[df['round'] == df['round'].max()].iloc[0]
            results.append({
                'num_clients': num_clients,
                'rouge1': last_round['rouge1'] * 100,  # Convert to percentage
                'rouge2': last_round['rouge2'] * 100,
                'rougeL': last_round['rougeL'] * 100,
                'loss': last_round['loss']
            })
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    return pd.DataFrame(results)

def plot_metrics_comparison(results_df, output_dir):
    """Plot comparison of ROUGE metrics across different numbers of clients."""
    if results_df.empty:
        print("No results to plot.")
        return
    
    # Sort by number of clients
    results_df = results_df.sort_values('num_clients')
    
    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Plot ROUGE-1, ROUGE-2, ROUGE-L
    plt.plot(results_df['num_clients'], results_df['rouge1'], 'o-', label='ROUGE-1', linewidth=2, markersize=8)
    plt.plot(results_df['num_clients'], results_df['rouge2'], 's-', label='ROUGE-2', linewidth=2, markersize=8)
    plt.plot(results_df['num_clients'], results_df['rougeL'], 'D-', label='ROUGE-L', linewidth=2, markersize=8)
    
    # Customize plot
    plt.title('ROUGE Scores vs Number of Clients', fontsize=16, pad=20)
    plt.xlabel('Number of Clients', fontsize=14)
    plt.ylabel('ROUGE Score (%)', fontsize=14)
    plt.xticks(results_df['num_clients'].unique())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Add data labels
    for i, row in results_df.iterrows():
        plt.text(row['num_clients'], row['rouge1']+0.2, f"{row['rouge1']:.1f}", ha='center', fontsize=9)
        plt.text(row['num_clients'], row['rouge2']-0.3, f"{row['rouge2']:.1f}", ha='center', fontsize=9)
        plt.text(row['num_clients'], row['rougeL']+0.2, f"{row['rougeL']:.1f}", ha='center', fontsize=9)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'clients_vs_rouge_comparison.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save a table of results
    table = results_df.to_markdown(index=False, floatfmt=".2f")
    with open(os.path.join(output_dir, 'clients_comparison_results.md'), 'w') as f:
        f.write("# Clients Comparison Results\n\n")
        f.write("ROUGE scores for different numbers of clients:\n\n")
        f.write(table)
    
    print(f"Results saved to {os.path.join(output_dir, 'clients_comparison_results.md')}")
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize comparison of ROUGE metrics across different numbers of clients')
    parser.add_argument('--results-dir', type=str, default='./experiment_results',
                      help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='./analysis_results',
                      help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Load and process results
    print("Loading results...")
    results_df = load_results(args.results_dir)
    
    if not results_df.empty:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate visualizations
        plot_metrics_comparison(results_df, args.output_dir)
    else:
        print("No valid results found to analyze.")
