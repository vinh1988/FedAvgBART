#!/usr/bin/env python3
"""
Federated DistilBART Training Visualization
-----------------------------------------

This script generates comprehensive visualizations for monitoring and analyzing the
performance of federated DistilBART models trained on the CNN/DailyMail dataset.

Key Features:
- Interactive visualization of training progress across federated rounds
- Detailed ROUGE metrics (F1, Precision, Recall) tracking
- Client-specific performance analysis
- Export capabilities for reports and presentations
"""

import os
import sys
import io
import logging
import argparse
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global style configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(
    style="whitegrid",
    palette="husl",
    rc={
        'figure.figsize': (20, 18),
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 18,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    }
)

# Custom color palette
COLOR_PALETTE = {
    'primary': '#3498db',    # Blue
    'success': '#2ecc71',    # Green
    'danger': '#e74c3c',     # Red
    'warning': '#f39c12',    # Orange
    'dark': '#2c3e50',       # Dark blue
    'light': '#ecf0f1',      # Light gray
    'text': '#2c3e50',       # Dark text
    'background': '#f8f9fa'  # Off-white
}

def load_metrics(results_dir: str = './results_distilbart_cnndm_federated') -> Tuple[pd.DataFrame, str]:
    """
    Load and preprocess the most recent metrics CSV file from federated training.
    
    Args:
        results_dir (str): Directory containing the metrics files. Defaults to 
                         './results_distilbart_cnndm_federated'.
    
    Returns:
        Tuple containing:
            - pd.DataFrame: Preprocessed metrics data
            - str: Path to the source metrics file
    
    Raises:
        FileNotFoundError: If no valid metrics files are found
        ValueError: If the metrics file is empty or malformed
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")
    
    # Find all matching metrics files
    metrics_files = list(results_path.glob('fed_distilbart_*_metrics_*.csv'))
    if not metrics_files:
        # Try alternative naming pattern
        metrics_files = list(results_path.glob('fed_distilbart_metrics_*.csv'))
        if not metrics_files:
            raise FileNotFoundError(
                f"No metrics files found in {results_path}. "
                "Expected pattern: 'fed_distilbart_*_metrics_*.csv'"
            )
    
    # Get the most recent file
    latest_file = max(metrics_files, key=os.path.getmtime)
    logger.info(f"Loading metrics from: {latest_file}")
    
    try:
        # Read and preprocess the file
        with open(latest_file, 'r') as f:
            # Skip comment lines and empty lines
            lines = [line for line in f if line.strip() and not line.startswith('#')]
        
        if not lines:
            raise ValueError(f"No valid data found in {latest_file}")
        
        # Read CSV data
        df = pd.read_csv(io.StringIO(''.join(lines)))
        
        # Validate required columns
        required_columns = {'round', 'client_id', 'phase', 'loss'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {latest_file}: {missing_columns}"
            )
        
        # Convert and validate data types
        numeric_columns = ['round', 'epoch', 'loss', 'f1', 'precision', 'recall']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process client IDs and global model flag
        df['client_id'] = df['client_id'].astype(str)
        df['is_global'] = df['client_id'].str.lower() == 'global'
        
        # Add derived metrics if not present
        if 'f1' not in df.columns and all(m in df.columns for m in ['precision', 'recall']):
            df['f1'] = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'] + 1e-10)
        
        # Sort by round for consistent plotting
        df = df.sort_values(['round', 'client_id']).reset_index(drop=True)
        
        return df, str(latest_file)
        
    except Exception as e:
        logger.error(f"Error loading metrics from {latest_file}: {str(e)}")
        raise
    
    # Print debug information
    print("\n=== Debug Info ===")
    print(f"Total rows: {len(df)}")
    print(f"Unique client_ids: {df['client_id'].unique()}")
    print(f"Global model rows: {df['is_global'].sum()}")
    print("First few rows of global evaluations:")
    print(df[df['is_global'] == True].head())
    print("\nFirst few rows of data:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("=================\n")
    
    return df, os.path.basename(latest_file)

def plot_metrics(
    df: pd.DataFrame,
    output_dir: str = './results_distilbart_cnndm_federated/plots',
    show_plots: bool = False,
    save_formats: list = ['png', 'pdf']
) -> Dict[str, str]:
    """
    Generate comprehensive visualizations for federated DistilBART training metrics.
    
    This function creates a multi-panel visualization including:
    - Training and validation loss curves
    - ROUGE metrics (F1, Precision, Recall)
    - Client performance analysis
    - Summary statistics
    
    Args:
        df: DataFrame containing the training metrics
        output_dir: Directory to save the output plots
        show_plots: Whether to display the plots interactively
        save_formats: List of file formats to save the plot in
        
    Returns:
        Dictionary mapping format to saved file path
        
    Raises:
        ValueError: If required data for plotting is missing
    """
    try:
        logger.info("Generating training metrics visualizations...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Filter and prepare data
        eval_metrics = df[df['is_global'] == True].sort_values('round')
        train_metrics = df[df['phase'] == 'train']
        
        if eval_metrics.empty or train_metrics.empty:
            raise ValueError("Insufficient data for visualization")
        
        # Initialize figure with custom grid
        fig = plt.figure(figsize=(22, 16))
        gs = fig.add_gridspec(
            nrows=2, ncols=2,
            width_ratios=[1.5, 1],
            height_ratios=[1, 1],
            hspace=0.4,
            wspace=0.3
        )
        
        # Main training plot (left side)
        ax1 = fig.add_subplot(gs[:, 0])
        plot_training_metrics(ax1, train_metrics, eval_metrics)
        
        # ROUGE F1 Score (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        plot_rouge_metrics(ax2, eval_metrics, metric='f1')
        
        # Precision/Recall (bottom right)
        ax3 = fig.add_subplot(gs[1, 1])
        plot_precision_recall(ax3, eval_metrics)
        
        # Add summary statistics
        add_summary_box(fig, eval_metrics, train_metrics)
        
        # Add footer with metadata
        add_footer(fig, train_metrics)
        
        # Final layout adjustments
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save in multiple formats
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"fed_distilbart_cnndm_metrics_{timestamp}"
        saved_files = {}
        
        for fmt in save_formats:
            try:
                filepath = output_path / f"{base_filename}.{fmt}"
                plt.savefig(
                    filepath,
                    dpi=300,
                    bbox_inches='tight',
                    facecolor=fig.get_facecolor(),
                    format=fmt
                )
                saved_files[fmt] = str(filepath)
                logger.info(f"Saved {fmt.upper()} visualization to: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to save {fmt.upper()} format: {str(e)}")
        
        if show_plots:
            plt.show()
        
        plt.close()
        return saved_files
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        raise

def plot_training_metrics(ax, train_metrics, eval_metrics):
    """Plot training and evaluation loss curves."""
    # Plot training loss per client
    sns.lineplot(
        ax=ax,
        data=train_metrics,
        x='round',
        y='loss',
        hue='client_id',
        style='client_id',
        markers=True,
        dashes=False,
        alpha=0.3,
        legend=False,
        linewidth=0.8,
        palette='husl'
    )
    
    # Plot global model evaluation loss
    if not eval_metrics.empty:
        sns.lineplot(
            ax=ax,
            data=eval_metrics,
            x='round',
            y='loss',
            color=COLOR_PALETTE['dark'],
            marker='o',
            markersize=8,
            linewidth=3,
            label='Global Model (Eval)'
        )
        
        # Add value annotations
        for x, y in zip(eval_metrics['round'], eval_metrics['loss']):
            ax.annotate(
                f'{y:.3f}',
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9,
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    fc='white',
                    alpha=0.8,
                    edgecolor='none'
                )
            )
    
    # Customize plot
    ax.set_title('Federated Training Progress', fontsize=18, pad=20, weight='bold')
    ax.set_xlabel('Federated Round', fontsize=12, labelpad=10)
    ax.set_ylabel('Loss', fontsize=12, labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add subtitle
    plt.figtext(
        0.12, 0.97,
        'DistilBART-CNN/DailyMail | Federated Learning',
        fontsize=14,
        style='italic',
        color=COLOR_PALETTE['text']
    )

def plot_rouge_metrics(ax, eval_metrics, metric='f1'):
    """Plot ROUGE metrics with confidence intervals if available."""
    if metric not in eval_metrics.columns:
        ax.text(0.5, 0.5, f'No {metric.upper()} data',
               ha='center', va='center', transform=ax.transAxes)
        return
    
    sns.lineplot(
        ax=ax,
        data=eval_metrics,
        x='round',
        y=metric,
        color=COLOR_PALETTE['success'],
        marker='o',
        markersize=8,
        linewidth=2.5,
        label=metric.upper()
    )
    
    # Add value annotations
    for x, y in zip(eval_metrics['round'], eval_metrics[metric]):
        ax.annotate(
            f'{y:.2f}',
            (x, y),
            textcoords="offset points",
            xytext=(0, 5),
            ha='center',
            fontsize=9,
            bbox=dict(
                boxstyle='round,pad=0.2',
                fc='white',
                alpha=0.8,
                edgecolor='none'
            )
        )
    
    # Customize plot
    ax.set_title(f'ROUGE-{metric.upper()} Score', fontsize=16, pad=15, weight='bold')
    ax.set_xlabel('Federated Round', fontsize=10, labelpad=8)
    ax.set_ylabel('Score', fontsize=10, labelpad=8)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def plot_precision_recall(ax, eval_metrics):
    """Plot precision and recall metrics."""
    if not all(m in eval_metrics.columns for m in ['precision', 'recall']):
        ax.text(0.5, 0.5, 'Missing precision/recall data',
               ha='center', va='center', transform=ax.transAxes)
        return
    
    # Plot precision
    sns.lineplot(
        ax=ax,
        data=eval_metrics,
        x='round',
        y='precision',
        color=COLOR_PALETTE['primary'],
        marker='o',
        markersize=6,
        label='Precision',
        linewidth=2.5
    )
    
    # Plot recall
    sns.lineplot(
        ax=ax,
        data=eval_metrics,
        x='round',
        y='recall',
        color=COLOR_PALETTE['danger'],
        marker='s',
        markersize=6,
        label='Recall',
        linewidth=2.5
    )
    
    # Add value annotations
    for x, y1, y2 in zip(eval_metrics['round'], 
                         eval_metrics['precision'], 
                         eval_metrics['recall']):
        ax.annotate(
            f'P:{y1:.1f}\nR:{y2:.1f}',
            (x, (y1 + y2) / 2),
            textcoords="offset points",
            xytext=(0, 0),
            ha='center',
            va='center',
            fontsize=7,
            bbox=dict(
                boxstyle='round,pad=0.2',
                fc='white',
                alpha=0.8,
                edgecolor='none'
            )
        )
    
    # Customize plot
    ax.set_title('ROUGE Precision & Recall', fontsize=16, pad=15, weight='bold')
    ax.set_xlabel('Federated Round', fontsize=10, labelpad=8)
    ax.set_ylabel('Score', fontsize=10, labelpad=8)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def add_summary_box(fig, eval_metrics, train_metrics):
    """Add a summary statistics box to the figure."""
    if eval_metrics.empty or train_metrics.empty:
        return
    
    try:
        final_round = int(eval_metrics['round'].max())
        final_metrics = eval_metrics[eval_metrics['round'] == final_round].iloc[0]
        
        # Calculate improvements
        if len(eval_metrics) > 1:
            first_metrics = eval_metrics.iloc[0]
            loss_improve = ((first_metrics['loss'] - final_metrics['loss']) / first_metrics['loss']) * 100
            f1_improve = ((final_metrics['f1'] - first_metrics['f1']) / (100 - first_metrics['f1'])) * 100
        else:
            loss_improve = f1_improve = 0
        
        stats_text = (
            f"Final Metrics (Round {final_round}):\n"
            f"• Loss: {final_metrics['loss']:.4f} ({loss_improve:+.1f}%)\n"
            f"• F1: {final_metrics['f1']:.2f} ({f1_improve:+.1f}%)\n"
            f"• Precision: {final_metrics['precision']:.2f}\n"
            f"• Recall: {final_metrics['recall']:.2f}\n"
            f"• Clients: {len(train_metrics['client_id'].unique())}\n"
            f"• Epochs: {int(train_metrics['epoch'].max())}"
        )
        
        # Add stats box
        props = dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.8,
            pad=0.5,
            edgecolor=COLOR_PALETTE['light']
        )
        
        fig.text(
            0.75, 0.15,
            stats_text,
            bbox=props,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            fontfamily='monospace',
            linespacing=1.5
        )
        
    except Exception as e:
        logger.warning(f"Could not generate summary box: {str(e)}")

def add_footer(fig, train_metrics):
    """Add footer with metadata to the figure."""
    try:
        num_clients = len(train_metrics['client_id'].unique()) if not train_metrics.empty else 0
        num_epochs = int(train_metrics['epoch'].max()) if not train_metrics.empty else 0
        
        footer_text = (
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{num_clients} clients | {num_epochs} epochs/client | "
            f"DistilBART-CNN/DailyMail"
        )
        
        plt.figtext(
            0.5, 0.01,
            footer_text,
            ha='center',
            fontsize=9,
            color=COLOR_PALETTE['text'],
            alpha=0.7
        )
    except Exception as e:
        logger.warning(f"Could not add footer: {str(e)}")

def print_metrics_summary(df):
    """Print a summary of the training and evaluation metrics."""
    print("\n" + "="*70)
    print("Federated DistilBART - CNN/DailyMail Training Summary")
    print("="*70)
    
    # Basic training info
    print(f"Total Rounds Completed: {df['round'].max() if not df.empty else 0}")
    
    # Training metrics
    train_df = df[df['phase'] == 'train']
    if not train_df.empty:
        print(f"Number of Clients: {len(train_df['client_id'].unique())}")
        print(f"Epochs per Client: {train_df['epoch'].max() + 1 if 'epoch' in train_df.columns else 'N/A'}")
        
        # Latest training metrics
        print("\nTraining Metrics (Final Round):")
        latest_round = train_df['round'].max()
        latest_train = train_df[train_df['round'] == latest_round]
        print(f"  - Average Loss: {latest_train['loss'].mean():.4f}")
    
    # Evaluation metrics
    eval_df = df[df['client_id'] == '-1']
    if not eval_df.empty:
        latest_eval = eval_df.iloc[-1]
        
        print("\nEvaluation Metrics (Final Round):")
        print(f"  - Loss: {latest_eval.get('loss', 'N/A'):.4f if 'loss' in latest_eval else 'N/A'}")
        print(f"  - ROUGE-L F1: {latest_eval.get('f1', 0):.2f}%")
        print(f"  - ROUGE-L Precision: {latest_eval.get('precision', 0):.2f}%")
        print(f"  - ROUGE-L Recall: {latest_eval.get('recall', 0):.2f}%")
        
        # Best metrics
        if 'f1' in eval_df.columns:
            best_rouge_idx = eval_df['f1'].idxmax()
            best_rouge = eval_df.loc[best_rouge_idx]
            print(f"\nBest ROUGE-L F1 (Round {best_rouge['round']}):")
            print(f"  - F1: {best_rouge['f1']:.2f}%")
            print(f"  - Precision: {best_rouge.get('precision', 0):.2f}%")
            print(f"  - Recall: {best_rouge.get('recall', 0):.2f}%")
    
    print("\n" + "="*70 + "\n")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize federated DistilBART training metrics.'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results_distilbart_cnndm_federated',
        help='Directory containing the metrics CSV files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results_distilbart_cnndm_federated/plots',
        help='Directory to save the output visualizations'
    )
    
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display the plots interactively (requires GUI)'
    )
    
    parser.add_argument(
        '--formats',
        type=str,
        nargs='+',
        default=['png', 'pdf'],
        help='Output formats (e.g., png pdf svg)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def main():
    """Main function to load metrics and generate visualizations."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Configure logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        
        logger.info(f"Loading metrics from: {args.results_dir}")
        
        # Load the metrics
        df, filename = load_metrics(args.results_dir)
        
        # Print summary
        print_metrics_summary(df)
        
        # Generate and save plots
        saved_files = plot_metrics(
            df=df,
            output_dir=args.output_dir,
            show_plots=args.show_plots,
            save_formats=args.formats
        )
        
        # Print saved file paths
        if saved_files:
            logger.info("\n" + "="*70)
            logger.info("Visualizations saved to:")
            for fmt, path in saved_files.items():
                logger.info(f"  • {fmt.upper()}: {path}")
            logger.info("="*70 + "\n")
        else:
            logger.warning("No visualizations were generated.")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=args.verbose)
        sys.exit(1)

if __name__ == "__main__":
    main()
