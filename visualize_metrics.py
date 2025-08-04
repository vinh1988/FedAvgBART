import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_metrics(metrics_file):
    # Load the metrics
    df = pd.read_csv(metrics_file)
    
    # Create output directory for plots
    os.makedirs("results/plots", exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (14, 8)
    
    # 1. Plot each metric in a separate figure
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    
    # Filter and process data
    phase_round = df[df['phase'].isin(['train', 'validation'])].groupby(['phase', 'round']).mean(numeric_only=True).reset_index()
    
    # Plot each metric separately
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for phase in ['train', 'validation']:
            phase_data = phase_round[phase_round['phase'] == phase]
            if not phase_data.empty:
                plt.plot(phase_data['round'], phase_data[metric], 
                        marker='o', label=f'{phase.capitalize()}', linewidth=2)
        
        plt.title(f'{metric.capitalize()} over Rounds')
        plt.xlabel('Round')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'results/plots/{metric}_over_rounds.png')
        plt.close()
    
    # 2. Plot all metrics in a single figure
    plt.figure(figsize=(14, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 2, i)
        for phase in ['train', 'validation']:
            phase_data = phase_round[phase_round['phase'] == phase]
            if not phase_data.empty:
                plt.plot(phase_data['round'], phase_data[metric], 
                        marker='o', label=f'{phase.capitalize()}')
        
        plt.title(f'{metric.capitalize()}')
        plt.xlabel('Round')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/plots/all_metrics.png')
    plt.close()
    
    # 3. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df[metrics].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title('Metrics Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('results/plots/metrics_correlation.png')
    plt.close()
    
    # 4. Client-wise metrics if multiple clients exist
    if 'client_id' in df.columns:
        train_df = df[df['phase'] == 'train']
        if len(train_df['client_id'].unique()) > 1:  # Only if we have multiple clients
            for metric in metrics:
                plt.figure(figsize=(10, 6))
                sns.barplot(data=train_df, x='client_id', y=metric, ci='sd')
                plt.title(f'Average {metric.capitalize()} by Client')
                plt.xlabel('Client ID')
                plt.ylabel(metric.capitalize())
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'results/plots/client_{metric}.png')
                plt.close()
    
    print("Plots saved to results/plots/ directory")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        metrics_file = sys.argv[1]
    else:
        # Get the most recent metrics file
        import glob
        files = glob.glob("results/training_metrics_*.csv")
        if not files:
            print("No metrics files found in results/ directory")
            sys.exit(1)
        metrics_file = max(files, key=os.path.getmtime)
    
    print(f"Visualizing metrics from: {metrics_file}")
    plot_metrics(metrics_file)