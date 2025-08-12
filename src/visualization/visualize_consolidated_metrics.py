import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import numpy as np
import argparse
from pathlib import Path
import json
from scipy import stats
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

# Set global style for publication quality figures
plt.style.use('seaborn-v0_8-paper')

# Configure matplotlib with improved defaults
mpl.rcParams.update({
    'text.usetex': False,  # Disable LaTeX text rendering
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.autolayout': True,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.edgecolor': '0.2',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6
})

# Color palette for consistent styling
PALETTE = sns.color_palette("colorblind")
STYLES = ['-', '--', ':', '-.']
MARKERS = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']

def load_metrics(csv_path: str) -> pd.DataFrame:
    """Load metrics from CSV file."""
    df = pd.read_csv(csv_path)
    return df

def plot_rouge_metrics(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot ROUGE metrics with confidence intervals across different numbers of clients.
    
    Args:
        df: DataFrame containing the metrics data
        output_dir: Directory to save the output figures
    """
    # Set up the figure with a modern style
    plt.style.use('seaborn-v0_8-pastel')
    
    # Create figure with a larger size and better DPI
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    
    # Define a more attractive color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    markers = ['o', 's', 'D']  # Circle, Square, Diamond
    
    # Define metrics and their pretty names
    rouge_metrics = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1']
    metric_names = {
        'rouge1_f1': 'ROUGE-1',
        'rouge2_f1': 'ROUGE-2',
        'rougeL_f1': 'ROUGE-L'
    }
    
    # Calculate mean and confidence intervals
    df_agg = df.groupby('num_clients')[rouge_metrics].agg(['mean', 'std', 'count'])
    
    # Create a background grid
    ax.set_facecolor('#f8f9fa')
    ax.grid(color='white', linestyle='-', linewidth=1, alpha=0.7)
    
    # Plot each metric with error bars
    for i, metric in enumerate(rouge_metrics):
        # Calculate 95% confidence interval
        y = df_agg[(metric, 'mean')]
        ci = 1.96 * df_agg[(metric, 'std')] / np.sqrt(df_agg[(metric, 'count')])
        
        # Plot the line with gradient color
        line, = ax.plot(df_agg.index, 
                       y,
                       marker=markers[i],
                       linestyle='-',
                       color=colors[i],
                       label=metric_names[metric],
                       markersize=10,
                       linewidth=2.5,
                       markerfacecolor='white',
                       markeredgewidth=1.5,
                       markeredgecolor=colors[i])
        
        # Add gradient fill under the line
        ax.fill_between(df_agg.index, 
                       y - ci, 
                       y + ci, 
                       color=colors[i],
                       alpha=0.15)
        
        # Add value annotations with a nice style
        for x_val, y_val, err in zip(df_agg.index, y, ci):
            ax.annotate(f"{y_val:.2f}", 
                       xy=(x_val, y_val + err + 0.01),
                       ha='center',
                       va='bottom',
                       fontsize=9,
                       fontweight='bold',
                       color=colors[i],
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white',
                               edgecolor=colors[i],
                               alpha=0.8,
                               linewidth=1))
    
    # Add statistical test results with better styling
    if len(df['num_clients'].unique()) > 1:
        groups = [df[df['num_clients'] == n]['rougeL_f1'] 
                 for n in df['num_clients'].unique() 
                 if not df[df['num_clients'] == n]['rougeL_f1'].isnull().all()]
        
        if len(groups) > 1 and all(len(g) > 1 for g in groups):
            f_val, p_val = stats.f_oneway(*groups)
            if p_val < 0.05:
                significance = "*" * min(3, sum(p_val < cutoff for cutoff in [0.05, 0.01, 0.001]))
                ax.text(0.02, 0.98, 
                       f"Significance: p = {p_val:.2e} {significance}",
                       transform=ax.transAxes,
                       ha='left',
                       va='top',
                       fontsize=10,
                       fontweight='bold',
                       bbox=dict(boxstyle='round',
                               facecolor='white',
                               edgecolor='#666666',
                               alpha=0.9,
                               linewidth=1.5))
    
    # Customize the plot appearance
    ax.set_xlabel('Number of Clients', fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontsize=12, labelpad=10, fontweight='bold')
    
    # Add a title with more style
    plt.title('ROUGE Scores by Number of Clients', 
             fontsize=16, pad=20, fontweight='bold',
             color='#2c3e50')
    
    # Customize the x and y axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Add a light grid
    ax.grid(True, linestyle='-', color='white', alpha=0.7)
    
    # Customize the legend with better positioning
    legend = ax.legend(loc='upper center',
                      bbox_to_anchor=(0.5, -0.15),  # Move legend slightly lower
                      ncol=3,
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      fontsize=10,  # Slightly smaller font
                      title='ROUGE Metrics',
                      title_fontsize=11)  # Slightly smaller title font
    
    # Add a border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#dddddd')
        spine.set_linewidth(1.5)
    
    # Add a subtle background pattern
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # Adjust layout with padding
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Add bottom padding for legend
    
    # Save in multiple formats
    os.makedirs(output_dir, exist_ok=True)
    for ext in ['.png', '.pdf']:
        output_path = os.path.join(output_dir, f'rouge_metrics{ext}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"ROUGE metrics plot saved to: {os.path.join(output_dir, 'rouge_metrics.*')}")
    print(f"ROUGE metrics plot saved to: {output_path}")

def plot_bleu_metrics(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot BLEU metrics with confidence intervals across different numbers of clients.
    
    Args:
        df: DataFrame containing the metrics data
        output_dir: Directory to save the output figures
    """
    # Set up the figure
    plt.figure(figsize=(9, 5))
    
    # Define BLEU metrics and their pretty names
    bleu_metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4']
    metric_names = {
        'bleu1': 'BLEU-1',
        'bleu2': 'BLEU-2',
        'bleu3': 'BLEU-3',
        'bleu4': 'BLEU-4'
    }
    
    # Calculate mean and confidence intervals for each client count
    client_counts = sorted(df['num_clients'].unique())
    bar_width = 0.2  # Width of the bars
    opacity = 0.8
    
    # Create positions for the bars
    index = np.arange(len(client_counts))
    
    # Plot each BLEU metric
    for i, metric in enumerate(bleu_metrics):
        # Calculate statistics for each client count
        stats = []
        for client_count in client_counts:
            subset = df[df['num_clients'] == client_count][metric]
            mean = subset.mean()
            std = subset.std()
            count = len(subset)
            ci = 1.96 * std / np.sqrt(count) if count > 1 else 0
            stats.append((mean, ci))
        
        means = [s[0] for s in stats]
        errors = [s[1] for s in stats]
        
        # Plot bars with error bars
        plt.bar(index + i*bar_width, 
                means, 
                bar_width,
                alpha=opacity,
                color=PALETTE[i % len(PALETTE)],
                yerr=errors,
                capsize=4,
                label=metric_names[metric])
        
        # Add value labels on top of bars
        for j, (mean, err) in enumerate(zip(means, errors)):
            plt.text(index[j] + i*bar_width, 
                    mean + err + 0.5,  # Position above the error bar
                    f'{mean:.1f} ± {err:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=45)
    
    plt.title('BLEU Scores by Number of Clients', fontsize=14, pad=15)
    plt.xlabel('Number of Clients', fontsize=12, labelpad=10)
    plt.ylabel('BLEU Score (%)', fontsize=12, labelpad=10)
    plt.xticks(index + bar_width * 1.5, client_counts)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.legend(fontsize=10, title='BLEU-n', title_fontsize=11)
    plt.tight_layout()
    
    # Save in multiple formats
    os.makedirs(output_dir, exist_ok=True)
    for ext in ['.png', '.pdf']:
        output_path = os.path.join(output_dir, f'bleu_metrics{ext}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"BLEU metrics plot saved to: {os.path.join(output_dir, 'bleu_metrics.*')}")
    print(f"BLEU metrics plot saved to: {os.path.join(output_dir, 'bleu_metrics.*')}")

def plot_training_loss(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot training loss across different numbers of clients.
    
    Args:
        df: DataFrame containing the metrics data
        output_dir: Directory to save the output figures
    """
    plt.figure(figsize=(8, 5))
    
    # Group by number of clients and calculate mean and std
    loss_stats = df.groupby('num_clients')['loss'].agg(['mean', 'std', 'count']).reset_index()
    
    # Plot with error bars
    plt.errorbar(loss_stats['num_clients'], 
                loss_stats['mean'],
                yerr=1.96 * loss_stats['std'] / np.sqrt(loss_stats['count']),
                marker='o',
                linestyle='-',
                color=PALETTE[0],
                markersize=8,
                linewidth=2,
                capsize=5,
                capthick=2,
                label='Mean Loss')
    
    # Add value annotations
    for _, row in loss_stats.iterrows():
        plt.text(row['num_clients'], 
                row['mean'] + 0.01, 
                f"{row['mean']:.3f}",
                ha='center',
                va='bottom',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', 
                         alpha=0.7,
                         edgecolor='none'))
    
    plt.title('Training Loss vs Number of Clients', fontsize=14, pad=15)
    plt.xlabel('Number of Clients', fontsize=12, labelpad=10)
    plt.ylabel('Loss', fontsize=12, labelpad=10)
    plt.xticks(sorted(df['num_clients'].unique()))
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save in multiple formats
    os.makedirs(output_dir, exist_ok=True)
    for ext in ['.png', '.pdf']:
        output_path = os.path.join(output_dir, f'training_loss{ext}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"Training loss plot saved to: {os.path.join(output_dir, 'training_loss.*')}")

def plot_contribution_metrics(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot client contribution metrics including Gini coefficient and CV of contributions.
    
    Args:
        df: DataFrame containing the metrics data
        output_dir: Directory to save the output figures
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.3)
    
    # --- Plot 1: Gini Coefficient ---
    if 'gini_coefficient' in df.columns:
        # Calculate statistics
        df_gini = df.groupby('num_clients')['gini_coefficient'].agg(['mean', 'std', 'count']).reset_index()
        
        # Plot with error bars
        ax1.errorbar(df_gini['num_clients'], 
                   df_gini['mean'],
                   yerr=1.96 * df_gini['std'] / np.sqrt(df_gini['count']),
                   marker=MARKERS[0],
                   linestyle=STYLES[0],
                   color=PALETTE[0],
                   markersize=6,
                   linewidth=1.5,
                   capsize=4,
                   label='Gini Coefficient')
        
        # Add value annotations
        for x, y in zip(df_gini['num_clients'], df_gini['mean']):
            ax1.text(x, y + 0.01, 
                   f"{y:.3f}", 
                   ha='center', 
                   va='bottom',
                   fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.2', 
                           fc='white', 
                           ec='none', 
                           alpha=0.7))
        
        ax1.set_xlabel('Number of Clients', fontsize=11, labelpad=8)
        ax1.set_ylabel('Gini Coefficient', fontsize=11, labelpad=8)
        ax1.set_title('Inequality in Client Contributions', fontsize=12, pad=12, fontweight='bold')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Add horizontal line at y=0 for reference
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    
    # --- Plot 2: CV of Contributions ---
    if 'cv_contribution' in df.columns:
        # Calculate statistics
        df_cv = df.groupby('num_clients')['cv_contribution'].agg(['mean', 'std', 'count']).reset_index()
        
        # Plot with error bars
        ax2.errorbar(df_cv['num_clients'], 
                   df_cv['mean'],
                   yerr=1.96 * df_cv['std'] / np.sqrt(df_cv['count']),
                   marker=MARKERS[1],
                   linestyle=STYLES[1],
                   color=PALETTE[1],
                   markersize=6,
                   linewidth=1.5,
                   capsize=4,
                   label='CV of Contributions')
        
        # Add value annotations
        for x, y in zip(df_cv['num_clients'], df_cv['mean']):
            ax2.text(x, y + 0.01, 
                   f"{y:.3f}", 
                   ha='center', 
                   va='bottom',
                   fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.2', 
                           fc='white', 
                           ec='none', 
                           alpha=0.7))
        
        ax2.set_xlabel('Number of Clients', fontsize=11, labelpad=8)
        ax2.set_ylabel('Coefficient of Variation', fontsize=11, labelpad=8)
        ax2.set_title('Variability in Client Contributions', fontsize=12, pad=12, fontweight='bold')
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Add overall title
    plt.suptitle('Client Contribution Analysis', fontsize=14, y=1.05, fontweight='bold')
    
    # Save in multiple formats
    os.makedirs(output_dir, exist_ok=True)
    for ext in ['.png', '.pdf', '.eps']:
        output_path = os.path.join(output_dir, f'contribution_metrics{ext}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"Contribution metrics plot saved to: {os.path.join(output_dir, 'contribution_metrics.*')}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Contribution metrics plot saved to: {output_path}")

def create_summary_table(df: pd.DataFrame, output_dir: str):
    """Create a summary table of metrics."""
    # Define all possible metrics with their display names
    all_metrics = {
        'num_clients': 'Clients',
        'rouge1_f1': 'ROUGE-1 F1',
        'rouge2_f1': 'ROUGE-2 F1',
        'rougeL_f1': 'ROUGE-L F1',
        'bleu1': 'BLEU-1',
        'bleu2': 'BLEU-2',
        'bleu3': 'BLEU-3',
        'bleu4': 'BLEU-4',
        'bertscore_f1': 'BERTScore F1',
        'loss': 'Loss',
        'gini_coefficient': 'Gini Coefficient',
        'cv_contribution': 'CV of Contributions',
        'mean_contribution': 'Mean Contribution',
        'min_contribution': 'Min Contribution',
        'max_contribution': 'Max Contribution'
    }
    
    # Only include metrics that exist in the dataframe
    available_metrics = {k: v for k, v in all_metrics.items() if k in df.columns}
    
    # Create a copy with only the available metrics
    df_summary = df[list(available_metrics.keys())].copy()
    
    # Format numbers for display
    for col in df_summary.columns:
        if col == 'Clients':
            continue
            
        if 'loss' in col.lower():
            df_summary[col] = df_summary[col].apply(lambda x: f"{x:.4f}")
        elif any(metric in col.lower() for metric in ['gini', 'cv']):
            df_summary[col] = df_summary[col].apply(lambda x: f"{x:.4f}")
        elif any(metric in col.lower() for metric in ['rouge', 'bleu', 'bertscore']):
            df_summary[col] = df_summary[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        else:
            df_summary[col] = df_summary[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'metrics_summary.csv')
    df_summary.to_csv(output_path, index=False)
    print(f"Metrics summary table saved to: {output_path}")

def main():
    """Main function to generate all visualizations."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate visualizations from consolidated metrics')
    parser.add_argument('--metrics-file', type=str, default='./experiment_results/analysis/consolidated_metrics.csv',
                      help='Path to the consolidated metrics CSV file')
    parser.add_argument('--output-dir', type=str, default='./experiment_results/analysis/plots',
                      help='Directory to save the output plots')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics
    print(f"Loading metrics from {args.metrics_file}...")
    df = load_metrics(args.metrics_file)
    
    # Generate plots
    print("Generating visualizations...")
    
    # Plot training metrics
    plot_training_loss(df, args.output_dir)
    
    # Plot evaluation metrics
    plot_rouge_metrics(df, args.output_dir)
    plot_bleu_metrics(df, args.output_dir)
    
    # Plot contribution metrics
    plot_contribution_metrics(df, args.output_dir)
    
    # Create summary table
    create_summary_table(df, args.output_dir)
    
    print("\nAll visualizations have been generated successfully!")
    print(f"Plots saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
