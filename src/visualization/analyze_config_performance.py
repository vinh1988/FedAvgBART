"""
Analyze and visualize the performance of different federated learning configurations.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Set the style for the plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

def load_metrics(csv_path: str) -> pd.DataFrame:
    """Load metrics from CSV file."""
    df = pd.read_csv(csv_path)
    return df

def plot_config_comparison(df: pd.DataFrame, output_dir: str):
    """Generate comparison plots for different configurations."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique client counts
    client_counts = sorted(df['num_clients'].unique())
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle('Configuration Performance Comparison', fontsize=16, y=1.02)
    
    # Plot 1: ROUGE Scores
    for metric in ['rouge1_f1', 'rouge2_f1', 'rougeL_f1']:
        sns.lineplot(ax=axes[0, 0], data=df, x='num_clients', y=metric, 
                    marker='o', label=metric)
    axes[0, 0].set_title('ROUGE Scores by Number of Clients')
    axes[0, 0].set_xlabel('Number of Clients')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend(title='Metric')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: BLEU Scores
    for i in range(1, 5):
        sns.lineplot(ax=axes[0, 1], data=df, x='num_clients', y=f'bleu{i}', 
                    marker='o', label=f'BLEU-{i}')
    axes[0, 1].set_title('BLEU Scores by Number of Clients')
    axes[0, 1].set_xlabel('Number of Clients')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend(title='Metric')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training Loss
    sns.lineplot(ax=axes[1, 0], data=df, x='num_clients', y='loss', 
                marker='o', color='red')
    axes[1, 0].set_title('Training Loss by Number of Clients')
    axes[1, 0].set_xlabel('Number of Clients')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: BERTScore
    sns.lineplot(ax=axes[1, 1], data=df, x='num_clients', y='bertscore_f1', 
                marker='o', color='purple')
    axes[1, 1].set_title('BERTScore F1 by Number of Clients')
    axes[1, 1].set_xlabel('Number of Clients')
    axes[1, 1].set_ylabel('BERTScore F1')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Client Contribution Metrics
    ax2 = axes[2, 0].twinx()
    sns.lineplot(ax=axes[2, 0], data=df, x='num_clients', y='gini_coefficient', 
                marker='o', color='green', label='Gini')
    sns.lineplot(ax=ax2, data=df, x='num_clients', y='cv_contribution', 
                marker='s', color='orange', label='CV')
    axes[2, 0].set_title('Client Contribution Metrics')
    axes[2, 0].set_xlabel('Number of Clients')
    axes[2, 0].set_ylabel('Gini Coefficient')
    ax2.set_ylabel('Coefficient of Variation')
    lines1, labels1 = axes[2, 0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2, 0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Client Participation
    participation = df[['num_clients', 'mean_contribution', 'min_contribution', 'max_contribution']].copy()
    participation_melted = pd.melt(participation, id_vars=['num_clients'], 
                                 value_vars=['mean_contribution', 'min_contribution', 'max_contribution'],
                                 var_name='metric', value_name='participation')
    sns.lineplot(ax=axes[2, 1], data=participation_melted, x='num_clients', 
                y='participation', hue='metric', marker='o')
    axes[2, 1].set_title('Client Participation Statistics')
    axes[2, 1].set_xlabel('Number of Clients')
    axes[2, 1].set_ylabel('Participation')
    axes[2, 1].legend(title='Metric')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'config_performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Configuration comparison plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize configuration performance.')
    parser.add_argument('--metrics-file', type=str, 
                       default='./experiment_results/analysis/consolidated_metrics.csv',
                       help='Path to the consolidated metrics CSV file')
    parser.add_argument('--output-dir', type=str, 
                       default='./experiment_results/analysis/plots',
                       help='Directory to save the output plots')
    
    args = parser.parse_args()
    
    print(f"Loading metrics from {args.metrics_file}...")
    df = load_metrics(args.metrics_file)
    
    print("Generating configuration comparison visualizations...")
    plot_config_comparison(df, args.output_dir)
    
    print("\nConfiguration analysis complete!")

if __name__ == "__main__":
    main()
