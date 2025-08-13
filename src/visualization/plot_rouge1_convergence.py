"""
Generate ROUGE-1 convergence plot across different client configurations.

This script creates a line plot showing how ROUGE-1 F1 scores evolve over training rounds
for different numbers of clients in the federated learning setup.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Set the style for the plots
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

def load_consolidated_metrics(csv_path: str) -> pd.DataFrame:
    """
    Load and preprocess the consolidated metrics CSV file.
    
    Args:
        csv_path: Path to the consolidated_metrics.csv file
        
    Returns:
        DataFrame containing the processed metrics
    """
    df = pd.read_csv(csv_path)
    
    # Ensure we have the required columns
    required_columns = ['num_clients', 'round', 'rouge1_f1']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    
    # Sort by client count and round
    df = df.sort_values(['num_clients', 'round'])
    
    return df

def plot_rouge1_convergence(df: pd.DataFrame, output_path: str):
    """
    Generate and save the ROUGE-1 convergence plot.
    
    Args:
        df: DataFrame containing the metrics data
        output_path: Path where to save the output plot
    """
    # Get unique client counts and sort them
    client_counts = sorted(df['num_clients'].unique())
    
    # Create a color palette with distinct colors for each client count
    palette = sns.color_palette("husl", n_colors=len(client_counts))
    
    # Create the figure with a larger size for better readability
    plt.figure(figsize=(12, 7))
    
    # Plot each client count
    for i, client_count in enumerate(client_counts):
        client_data = df[df['num_clients'] == client_count]
        
        # Skip if not enough data points
        if len(client_data) < 2:
            continue
            
        plt.plot(
            client_data['round'], 
            client_data['rouge1_f1'], 
            label=f'{client_count} Clients',
            color=palette[i],
            linewidth=2.5,
            marker='o',
            markersize=6,
            markevery=max(1, len(client_data) // 5)  # Show some markers but not too many
        )
        
        # Add final value annotation
        final_round = client_data['round'].max()
        final_score = client_data[client_data['round'] == final_round]['rouge1_f1'].values[0]
        plt.annotate(
            f'{final_score:.1f}%', 
            xy=(final_round, final_score),
            xytext=(5, 0), 
            textcoords='offset points',
            ha='left', 
            va='center',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, edgecolor='none'),
            color=palette[i]
        )
    
    # Customize the plot
    plt.title('ROUGE-1 Score Convergence by Client Count', fontsize=14, pad=15)
    plt.xlabel('Training Round', fontsize=12, labelpad=10)
    plt.ylabel('ROUGE-1 F1 Score (%)', fontsize=12, labelpad=10)
    
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Number of Clients', title_fontsize=11, fontsize=10, 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    # Save the figure with high DPI for publication quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate ROUGE-1 convergence plot')
    parser.add_argument('--input', type=str, 
                       default=os.path.join(os.path.dirname(__file__), 
                                          '../../experiment_results/analysis/consolidated_metrics.csv'),
                       help='Path to consolidated_metrics.csv')
    parser.add_argument('--output', type=str, 
                       default=os.path.join(os.path.dirname(__file__),
                                          '../../experiment_results/analysis/plots/rouge1_convergence.png'),
                       help='Output path for the plot')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load and process the data
        print(f"Loading metrics from: {args.input}")
        df = load_consolidated_metrics(args.input)
        
        # Generate and save the plot
        print(f"Generating plot and saving to: {args.output}")
        plot_rouge1_convergence(df, args.output)
        
        print("ROUGE-1 convergence plot generated successfully!")
        
    except Exception as e:
        print(f"Error generating plot: {str(e)}")
        raise

if __name__ == "__main__":
    main()
