import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_contribution_metrics(output_dir):
    """Generate plots for client contribution metrics."""
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, 'contribution_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data
    try:
        with open(os.path.join(output_dir, 'round_metrics.json'), 'r') as f:
            round_metrics = json.load(f)
        client_metrics = pd.read_csv(os.path.join(output_dir, 'client_metrics.csv'))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Gini coefficient over rounds
    plt.figure()
    rounds = [m['round'] for m in round_metrics]
    gini_values = [m['gini'] for m in round_metrics]
    
    plt.plot(rounds, gini_values, marker='o', color='#2ecc71', linewidth=2, markersize=8)
    plt.title('Gini Coefficient of Client Contributions Over Rounds', fontsize=14, pad=20)
    plt.xlabel('Round', fontsize=12, labelpad=10)
    plt.ylabel('Gini Coefficient', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'gini_over_rounds.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Contribution distribution by round
    plt.figure()
    sns.boxplot(x='round', y='contribution', data=client_metrics, 
                palette='viridis', showfliers=False)
    plt.title('Distribution of Client Contributions by Round', fontsize=14, pad=20)
    plt.xlabel('Round', fontsize=12, labelpad=10)
    plt.ylabel('Contribution (L2 Norm)', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'contribution_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Client participation
    plt.figure()
    participation = client_metrics.groupby('client_id').size()
    participation.plot(kind='bar', color=sns.color_palette('viridis', len(participation)))
    plt.title('Client Participation Count', fontsize=14, pad=20)
    plt.xlabel('Client ID', fontsize=12, labelpad=10)
    plt.ylabel('Number of Rounds Participated', fontsize=12, labelpad=10)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'client_participation.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Contribution vs Data Size
    plt.figure()
    sns.scatterplot(data=client_metrics, x='data_size', y='contribution', 
                   hue='client_id', palette='viridis', s=100, alpha=0.7)
    plt.title('Client Contribution vs Data Size', fontsize=14, pad=20)
    plt.xlabel('Data Size (samples)', fontsize=12, labelpad=10)
    plt.ylabel('Contribution (L2 Norm)', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Client ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'contribution_vs_data_size.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Heatmap of client contributions
    plt.figure(figsize=(12, 8))
    client_pivot = client_metrics.pivot(index='client_id', columns='round', values='contribution')
    sns.heatmap(client_pivot, cmap='viridis', annot=True, fmt='.2f', 
                cbar_kws={'label': 'Contribution (L2 Norm)'})
    plt.title('Client Contributions Heatmap', fontsize=14, pad=20)
    plt.xlabel('Round', fontsize=12, labelpad=10)
    plt.ylabel('Client ID', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'contribution_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to: {plots_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize client contribution metrics')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory containing the experiment results')
    
    args = parser.parse_args()
    plot_contribution_metrics(args.output_dir)
