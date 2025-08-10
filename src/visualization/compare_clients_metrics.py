import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_metrics_for_clients(base_dir, min_clients=2, max_clients=10):
    """Load metrics for different client configurations."""
    all_metrics = {}
    
    for num_clients in range(min_clients, max_clients + 1):
        # Find all metrics files for this number of clients
        metrics_files = glob.glob(os.path.join(
            base_dir, f"clients_{num_clients}", "fed_distilbart_cnndm_metrics_*.json"
        ))
        
        if not metrics_files:
            print(f"No metrics files found for {num_clients} clients")
            continue
            
        # Use the most recent metrics file
        latest_file = max(metrics_files, key=os.path.getctime)
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
                all_metrics[num_clients] = {
                    'file': latest_file,
                    'metrics': data.get('global_metrics', {})
                }
        except Exception as e:
            print(f"Error loading {latest_file}: {e}")
    
    return all_metrics

def extract_rouge_metrics(metrics_data):
    """Extract ROUGE metrics from the metrics data."""
    rouge_metrics = {}
    
    for round_num, metrics in metrics_data.items():
        if 'phase' in metrics and metrics['phase'] == 'evaluation':
            round_metrics = {}
            
            # Handle both old and new ROUGE metric names
            if all(f'rouge{i}' in metrics for i in [1, 2, 'L']):
                round_metrics.update({
                    'ROUGE-1': metrics['rouge1'] * 100,
                    'ROUGE-2': metrics['rouge2'] * 100,
                    'ROUGE-L': metrics['rougeL'] * 100
                })
            elif all(f'rouge{i}_f1' in metrics for i in [1, 2, 'L']):
                round_metrics.update({
                    'ROUGE-1': metrics['rouge1_f1'] * 100,
                    'ROUGE-2': metrics['rouge2_f1'] * 100,
                    'ROUGE-L': metrics['rougeL_f1'] * 100
                })
            
            if round_metrics:  # Only add if we found ROUGE metrics
                rouge_metrics[int(round_num)] = round_metrics
    
    return rouge_metrics

def plot_rouge_comparison(metrics_data, output_dir):
    """Plot comparison of ROUGE metrics across different client configurations."""
    # Prepare data for plotting
    plot_data = []
    
    for num_clients, data in metrics_data.items():
        rouge_metrics = extract_rouge_metrics(data['metrics'])
        
        for round_num, metrics in rouge_metrics.items():
            for metric_name, value in metrics.items():
                plot_data.append({
                    'Clients': num_clients,
                    'Round': round_num,
                    'Metric': metric_name,
                    'Score': value
                })
    
    if not plot_data:
        print("No ROUGE metrics found for plotting.")
        return
    
    df = pd.DataFrame(plot_data)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create a single figure with all ROUGE metrics together
    plt.figure(figsize=(14, 8))
    
    # Create a line plot with all ROUGE metrics
    palette = sns.color_palette("husl", 3)  # One color per metric
    
    # Create a grid of subplots: one for each client count
    client_counts = sorted(df['Clients'].unique())
    n_cols = min(3, len(client_counts))
    n_rows = (len(client_counts) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    fig.suptitle('ROUGE Metrics Comparison Across Client Configurations', 
                fontsize=16, y=1.02)
    
    # Flatten axes if needed
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each client configuration
    for idx, num_clients in enumerate(client_counts):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[col] if n_rows == 1 else axes[row]
        
        # Filter data for this client count
        client_df = df[df['Clients'] == num_clients]
        
        # Plot each metric
        for i, metric in enumerate(['ROUGE-1', 'ROUGE-2', 'ROUGE-L']):
            metric_df = client_df[client_df['Metric'] == metric].sort_values('Round')
            if not metric_df.empty:
                ax.plot(metric_df['Round'], metric_df['Score'], 
                       marker='o', markersize=6, linewidth=2,
                       label=metric, color=palette[i])
                
                # Add value annotations for the last point
                last_point = metric_df.iloc[-1]
                ax.annotate(f'{last_point["Score"]:.1f}%', 
                           xy=(last_point['Round'], last_point['Score']),
                           xytext=(5, 0), textcoords='offset points',
                           ha='left', va='center', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   fc='white', alpha=0.8))
        
        ax.set_title(f'{num_clients} Clients', fontsize=12, pad=10)
        ax.set_xlabel('Training Round', fontsize=10, labelpad=8)
        ax.set_ylabel('Score (%)', fontsize=10, labelpad=8)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only add legend to the first subplot
        if idx == 0:
            ax.legend(fontsize=9, loc='upper left')
    
    # Hide any empty subplots
    for idx in range(len(client_counts), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows > 1 and n_cols > 1:
            axes[row, col].axis('off')
        elif n_rows == 1:
            axes[col].axis('off')
        else:
            axes[row].axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'rouge_metrics_comparison_combined.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined ROUGE metrics comparison plot saved to: {output_path}")
    
    # Also create the line plot comparison
    plot_rouge_line_comparison(df, output_dir)

def plot_rouge_line_comparison(df, output_dir):
    """Create line plots comparing ROUGE metrics across client configurations."""
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 10))
    
    # Create a single figure with all metrics
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors and markers for each metric
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    colors = sns.color_palette("husl", len(metrics))
    markers = ['o', 's', '^']
    
    # Plot each metric with a different color and marker
    for i, metric in enumerate(metrics):
        metric_df = df[df['Metric'] == metric].copy()
        if metric_df.empty:
            continue
            
        # Pivot to get scores by client and round
        pivot_df = metric_df.pivot(index='Round', columns='Clients', values='Score')
        
        # Plot each client count with the same color but different line styles
        for j, client_count in enumerate(sorted(pivot_df.columns)):
            line_style = ['-', '--', ':', '-.'][j % 4]  # Cycle through line styles
            line = ax.plot(pivot_df.index, pivot_df[client_count], 
                          marker=markers[i], markersize=6, linewidth=1.5,
                          linestyle=line_style, color=colors[i],
                          alpha=0.8, 
                          label=f'{metric} - {client_count} Clients' if j == 0 else "")
            
            # Add value annotation for the last point
            last_val = pivot_df[client_count].iloc[-1]
            ax.annotate(f'{last_val:.1f}%', 
                       xy=(pivot_df.index[-1], last_val),
                       xytext=(5, 0), textcoords='offset points',
                       ha='left', va='center', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', 
                               fc='white', alpha=0.8))
    
    # Customize the plot
    ax.set_title('ROUGE Metrics Comparison Across Client Configurations', 
                fontsize=14, pad=20)
    ax.set_xlabel('Training Round', fontsize=12, labelpad=10)
    ax.set_ylabel('F1 Score (%)', fontsize=12, labelpad=10)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    
    # Metric legend (colors and markers)
    metric_legend = [Line2D([0], [0], color=colors[i], marker=markers[i], 
                          linestyle='-', label=metric,
                          markersize=8, linewidth=2) 
                    for i, metric in enumerate(metrics) if not df[df['Metric'] == metric].empty]
    
    # Client count legend (line styles)
    client_counts = sorted(df['Clients'].unique())
    line_styles = ['-', '--', ':', '-.']
    client_legend = [Line2D([0], [0], color='gray', 
                           linestyle=line_styles[i % len(line_styles)],
                           label=f'{count} Clients',
                           linewidth=1.5) 
                    for i, count in enumerate(client_counts)]
    
    # Add legends to the plot
    legend1 = ax.legend(handles=metric_legend, title='Metrics',
                       loc='upper left', bbox_to_anchor=(1.02, 1),
                       borderaxespad=0.)
    ax.add_artist(legend1)
    
    ax.legend(handles=client_legend, title='Number of Clients',
             loc='lower left', bbox_to_anchor=(1.02, 0),
             borderaxespad=0.)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'rouge_metrics_line_comparison_combined.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined ROUGE metrics line comparison plot saved to: {output_path}")
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'rouge_metrics_line_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROUGE metrics line comparison plot saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare ROUGE metrics across different client configurations.'
    )
    parser.add_argument('--base-dir', type=str, default='./experiment_results',
                       help='Base directory containing client experiment results')
    parser.add_argument('--output-dir', type=str, default='./experiment_results/analysis',
                       help='Directory to save the comparison plots')
    parser.add_argument('--min-clients', type=int, default=2,
                       help='Minimum number of clients to include')
    parser.add_argument('--max-clients', type=int, default=10,
                       help='Maximum number of clients to include')
    
    args = parser.parse_args()
    
    # Load metrics for all client configurations
    metrics_data = load_metrics_for_clients(
        args.base_dir, 
        min_clients=args.min_clients, 
        max_clients=args.max_clients
    )
    
    if not metrics_data:
        print("No metrics data found for the specified client configurations.")
    else:
        # Create comparison plots
        plot_rouge_comparison(metrics_data, args.output_dir)
