import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def plot_training_metrics(metrics_file, output_dir):
    """Plot training metrics including ROUGE and BLEU scores from the experiment results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics data
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    # Extract global metrics
    global_metrics = data.get('global_metrics', {})
    if not global_metrics:
        print("No global metrics found in the metrics file.")
        return
    
    # Prepare data for plotting
    metrics_data = []
    
    for round_num, metrics in global_metrics.items():
        if 'phase' in metrics and metrics['phase'] == 'evaluation':
            round_data = {'Round': int(round_num)}
            
            # ROUGE metrics
            if 'rouge1' in metrics and 'rouge2' in metrics and 'rougeL' in metrics:
                round_data.update({
                    'ROUGE-1': metrics['rouge1'] * 100,
                    'ROUGE-2': metrics['rouge2'] * 100,
                    'ROUGE-L': metrics['rougeL'] * 100
                })
            elif 'rouge1_f1' in metrics and 'rouge2_f1' in metrics and 'rougeL_f1' in metrics:
                round_data.update({
                    'ROUGE-1': metrics['rouge1_f1'] * 100,
                    'ROUGE-2': metrics['rouge2_f1'] * 100,
                    'ROUGE-L': metrics['rougeL_f1'] * 100
                })
            
            # BLEU metrics
            for n in range(1, 5):
                bleu_key = f'bleu{n}'
                if bleu_key in metrics:
                    round_data[f'BLEU-{n}'] = metrics[bleu_key] * 100
                elif f'bleu_{n}' in metrics:
                    round_data[f'BLEU-{n}'] = metrics[f'bleu_{n}'] * 100
            
            # Other metrics
            if 'loss' in metrics:
                round_data['Loss'] = metrics['loss']
            
            if 'bertscore_f1' in metrics:
                round_data['BERTScore'] = metrics['bertscore_f1'] * 100
            
            metrics_data.append(round_data)
    
    if not metrics_data:
        print("No evaluation rounds with metrics found.")
        return
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame(metrics_data).sort_values('Round')
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Metrics Over Training Rounds', fontsize=16, y=1.02)
    
    # Plot 1: ROUGE metrics
    if all(m in df.columns for m in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']):
        ax = axes[0, 0]
        for metric in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']:
            ax.plot(df['Round'], df[metric], marker='o', label=metric, linewidth=2, markersize=6)
        
        ax.set_title('ROUGE F1 Scores', fontsize=12, pad=10)
        ax.set_xlabel('Round', fontsize=10, labelpad=8)
        ax.set_ylabel('Score (%)', fontsize=10, labelpad=8)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add value annotations for the last point
        for metric in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']:
            last_val = df[metric].iloc[-1]
            ax.annotate(f'{last_val:.1f}%', 
                       xy=(df['Round'].iloc[-1], last_val),
                       xytext=(5, 0), textcoords='offset points',
                       ha='left', va='center', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    
    # Plot 2: BLEU metrics
    bleu_cols = [col for col in df.columns if col.startswith('BLEU-')]
    if bleu_cols:
        ax = axes[0, 1]
        for metric in sorted(bleu_cols):
            ax.plot(df['Round'], df[metric], marker='s', label=metric, linewidth=2, markersize=6)
        
        ax.set_title('BLEU-n Scores', fontsize=12, pad=10)
        ax.set_xlabel('Round', fontsize=10, labelpad=8)
        ax.set_ylabel('Score (%)', fontsize=10, labelpad=8)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add value annotations for the last point of each BLEU score
        for metric in sorted(bleu_cols):
            last_val = df[metric].iloc[-1]
            ax.annotate(f'{metric}: {last_val:.1f}%', 
                       xy=(df['Round'].iloc[-1], last_val),
                       xytext=(5, 0), textcoords='offset points',
                       ha='left', va='center', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    
    # Plot 3: BERTScore if available
    if 'BERTScore' in df.columns:
        ax = axes[1, 0]
        ax.plot(df['Round'], df['BERTScore'], marker='^', color='purple', 
                label='BERTScore F1', linewidth=2, markersize=8)
        
        # Add the last value annotation
        last_val = df['BERTScore'].iloc[-1]
        ax.annotate(f'F1: {last_val:.1f}%', 
                   xy=(df['Round'].iloc[-1], last_val),
                   xytext=(5, 0), textcoords='offset points',
                   ha='left', va='center', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        
        ax.set_title('BERTScore', fontsize=12, pad=10)
        ax.set_xlabel('Round', fontsize=10, labelpad=8)
        ax.set_ylabel('F1 Score (%)', fontsize=10, labelpad=8)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Loss if available
    if 'Loss' in df.columns:
        ax = axes[1, 1]
        ax.plot(df['Round'], df['Loss'], marker='d', color='red', 
                label='Loss', linewidth=2, markersize=8)
        
        # Add the last value annotation
        last_val = df['Loss'].iloc[-1]
        ax.annotate(f'{last_val:.3f}', 
                   xy=(df['Round'].iloc[-1], last_val),
                   xytext=(5, 0), textcoords='offset points',
                   ha='left', va='center', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        
        ax.set_title('Training Loss', fontsize=12, pad=10)
        ax.set_xlabel('Round', fontsize=10, labelpad=8)
        ax.set_ylabel('Loss', fontsize=10, labelpad=8)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training metrics plot saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot training metrics from experiment results')
    parser.add_argument('--metrics-file', type=str, required=True,
                       help='Path to the metrics JSON file')
    parser.add_argument('--output-dir', type=str, default='./plots',
                       help='Directory to save the plots (default: ./plots)')
    
    args = parser.parse_args()
    plot_training_metrics(args.metrics_file, args.output_dir)
