import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import numpy as np

# Set the style for the plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

def load_metrics(csv_path: str) -> pd.DataFrame:
    """Load metrics from CSV file."""
    df = pd.read_csv(csv_path)
    return df

def plot_rouge_metrics(df: pd.DataFrame, output_dir: str):
    """Plot ROUGE metrics across different client configurations."""
    plt.figure(figsize=(12, 6))
    
    # Melt the dataframe for easier plotting
    rouge_metrics = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1']
    df_melted = df.melt(id_vars=['num_clients'], 
                        value_vars=rouge_metrics,
                        var_name='Metric', 
                        value_name='Score')
    
    # Clean up metric names for display
    df_melted['Metric'] = df_melted['Metric'].str.replace('_f1', '').str.upper()
    
    # Create the plot
    ax = sns.lineplot(data=df_melted, x='num_clients', y='Score', 
                     hue='Metric', marker='o', markersize=8, linewidth=2)
    
    # Add value annotations
    for metric in df_melted['Metric'].unique():
        metric_data = df_melted[df_melted['Metric'] == metric]
        for _, row in metric_data.iterrows():
            ax.text(row['num_clients'], row['Score'] + 0.5, 
                   f"{row['Score']:.1f}", 
                   ha='center', va='bottom', fontsize=8)
    
    plt.title('ROUGE Metrics vs Number of Clients', fontsize=14, pad=15)
    plt.xlabel('Number of Clients', fontsize=12, labelpad=10)
    plt.ylabel('F1 Score (%)', fontsize=12, labelpad=10)
    plt.xticks(df['num_clients'].unique())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='ROUGE Metric', fontsize=10, title_fontsize=11)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'rouge_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROUGE metrics plot saved to: {output_path}")

def plot_bleu_metrics(df: pd.DataFrame, output_dir: str):
    """Plot BLEU metrics across different client configurations."""
    plt.figure(figsize=(12, 6))
    
    # Melt the dataframe for easier plotting
    bleu_metrics = [f'bleu{i}' for i in range(1, 5)]
    df_melted = df.melt(id_vars=['num_clients'], 
                        value_vars=bleu_metrics,
                        var_name='Metric', 
                        value_name='Score')
    
    # Clean up metric names for display
    df_melted['Metric'] = df_melted['Metric'].str.upper()
    
    # Create the plot
    ax = sns.lineplot(data=df_melted, x='num_clients', y='Score', 
                     hue='Metric', marker='o', markersize=8, linewidth=2)
    
    # Add value annotations for the last point of each line
    for metric in df_melted['Metric'].unique():
        metric_data = df_melted[df_melted['Metric'] == metric]
        last_point = metric_data.iloc[-1]
        ax.text(last_point['num_clients'], last_point['Score'] + 0.3, 
               f"{last_point['Score']:.1f}", 
               ha='center', va='bottom', fontsize=8)
    
    plt.title('BLEU Scores vs Number of Clients', fontsize=14, pad=15)
    plt.xlabel('Number of Clients', fontsize=12, labelpad=10)
    plt.ylabel('BLEU Score (%)', fontsize=12, labelpad=10)
    plt.xticks(df['num_clients'].unique())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='BLEU-n', fontsize=10, title_fontsize=11)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'bleu_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"BLEU metrics plot saved to: {output_path}")

def plot_bertscore_and_loss(df: pd.DataFrame, output_dir: str):
    """Plot BERTScore F1 and Loss metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot BERTScore F1
    sns.lineplot(data=df, x='num_clients', y='bertscore_f1', 
                marker='o', markersize=8, linewidth=2, ax=ax1, color='#2ecc71')
    
    # Add value annotations for BERTScore
    for _, row in df.iterrows():
        ax1.text(row['num_clients'], row['bertscore_f1'] + 0.1, 
                f"{row['bertscore_f1']:.1f}", 
                ha='center', va='bottom', fontsize=8)
    
    ax1.set_title('BERTScore F1 vs Number of Clients', fontsize=14, pad=15)
    ax1.set_xlabel('Number of Clients', fontsize=12, labelpad=10)
    ax1.set_ylabel('BERTScore F1 (%)', fontsize=12, labelpad=10)
    ax1.set_xticks(df['num_clients'].unique())
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Loss
    sns.lineplot(data=df, x='num_clients', y='loss', 
                marker='o', markersize=8, linewidth=2, ax=ax2, color='#e74c3c')
    
    # Add value annotations for Loss
    for _, row in df.iterrows():
        ax2.text(row['num_clients'], row['loss'] + 0.05, 
                f"{row['loss']:.2f}", 
                ha='center', va='bottom', fontsize=8)
    
    ax2.set_title('Training Loss vs Number of Clients', fontsize=14, pad=15)
    ax2.set_xlabel('Number of Clients', fontsize=12, labelpad=10)
    ax2.set_ylabel('Loss', fontsize=12, labelpad=10)
    ax2.set_xticks(df['num_clients'].unique())
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'bertscore_and_loss.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"BERTScore and Loss plot saved to: {output_path}")

def plot_contribution_metrics(df: pd.DataFrame, output_dir: str):
    """Plot client contribution metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Gini Coefficient
    sns.lineplot(data=df, x='num_clients', y='gini_coefficient', 
                marker='o', markersize=8, linewidth=2, ax=ax1, color='#9b59b6')
    
    # Add value annotations for Gini Coefficient
    for _, row in df.iterrows():
        ax1.text(row['num_clients'], row['gini_coefficient'] + 0.001, 
                f"{row['gini_coefficient']:.4f}", 
                ha='center', va='bottom', fontsize=8)
    
    ax1.set_title('Gini Coefficient vs Number of Clients', fontsize=14, pad=15)
    ax1.set_xlabel('Number of Clients', fontsize=12, labelpad=10)
    ax1.set_ylabel('Gini Coefficient', fontsize=12, labelpad=10)
    ax1.set_xticks(df['num_clients'].unique())
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Coefficient of Variation
    sns.lineplot(data=df, x='num_clients', y='cv_contribution', 
                marker='o', markersize=8, linewidth=2, ax=ax2, color='#3498db')
    
    # Add value annotations for CV
    for _, row in df.iterrows():
        ax2.text(row['num_clients'], row['cv_contribution'] + 0.001, 
                f"{row['cv_contribution']:.4f}", 
                ha='center', va='bottom', fontsize=8)
    
    ax2.set_title('Coefficient of Variation vs Number of Clients', fontsize=14, pad=15)
    ax2.set_xlabel('Number of Clients', fontsize=12, labelpad=10)
    ax2.set_ylabel('CV of Contributions', fontsize=12, labelpad=10)
    ax2.set_xticks(df['num_clients'].unique())
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'contribution_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Contribution metrics plot saved to: {output_path}")

def create_summary_table(df: pd.DataFrame, output_dir: str):
    """Create a summary table of metrics."""
    # Select and reorder columns
    metrics = ['num_clients', 'rouge1_f1', 'rouge2_f1', 'rougeL_f1', 
              'bleu1', 'bleu2', 'bleu3', 'bleu4', 'bertscore_f1', 'loss',
              'gini_coefficient', 'cv_contribution']
    
    # Format the dataframe
    df_summary = df[metrics].copy()
    
    # Rename columns for display
    column_renames = {
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
        'gini_coefficient': 'Gini Coef',
        'cv_contribution': 'CV of Contr.'
    }
    
    df_summary = df_summary.rename(columns=column_renames)
    
    # Format numbers for display
    for col in df_summary.columns[1:]:  # Skip 'Clients' column
        if 'loss' in col.lower():
            df_summary[col] = df_summary[col].apply(lambda x: f"{x:.4f}")
        elif any(metric in col.lower() for metric in ['gini', 'cv']):
            df_summary[col] = df_summary[col].apply(lambda x: f"{x:.4f}")
        else:
            df_summary[col] = df_summary[col].apply(lambda x: f"{x:.2f}")
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'metrics_summary.csv')
    df_summary.to_csv(output_path, index=False)
    print(f"Metrics summary table saved to: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize consolidated metrics.')
    parser.add_argument('--metrics-file', type=str, 
                       default='./experiment_results/analysis/consolidated_metrics.csv',
                       help='Path to the consolidated metrics CSV file')
    parser.add_argument('--output-dir', type=str, 
                       default='./experiment_results/analysis/plots',
                       help='Directory to save the output plots')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics
    print(f"Loading metrics from {args.metrics_file}...")
    df = load_metrics(args.metrics_file)
    
    # Generate plots
    print("Generating visualizations...")
    plot_rouge_metrics(df, args.output_dir)
    plot_bleu_metrics(df, args.output_dir)
    plot_bertscore_and_loss(df, args.output_dir)
    plot_contribution_metrics(df, args.output_dir)
    create_summary_table(df, args.output_dir)
    
    print("\nAll visualizations have been generated successfully!")
    print(f"Plots saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
