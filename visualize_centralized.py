import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import sys

def plot_metrics(metrics_file, output_dir="results/plots_centralized"):
    """
    Plot training metrics from centralized training.
    
    Args:
        metrics_file (str): Path to the metrics CSV file
        output_dir (str): Directory to save the plots
    """
    # Load the metrics
    df = pd.read_csv(metrics_file)
    
    # Create output directory for plots
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (14, 8)
    
    # 1. Plot each metric in a separate figure
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    
    # Filter and process data
    train_val_df = df[df['phase'].isin(['train', 'validation'])].copy()
    
    # For validation data, use the same epoch as the corresponding training data
    # This assumes validation is done right after each training epoch
    train_epochs = train_val_df[train_val_df['phase'] == 'train']['epoch'].values
    val_epochs = train_val_df[train_val_df['phase'] == 'validation']['epoch'].values
    
    # If we have matching train/val epochs, use them directly
    if len(train_epochs) == len(val_epochs):
        train_val_df.loc[train_val_df['phase'] == 'validation', 'epoch'] = train_epochs
    
    # Plot each metric separately
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for phase in ['train', 'validation']:
            phase_data = train_val_df[train_val_df['phase'] == phase].sort_values('epoch')
            if not phase_data.empty:
                x_values = phase_data['epoch'] if phase == 'train' else phase_data['epoch'] + 0.1  # Slight offset for visibility
                plt.plot(x_values, phase_data[metric], 
                        marker='o' if phase == 'train' else 's',  # Different markers for train/val
                        label=f'{phase.capitalize()}', 
                        linewidth=2,
                        markersize=6)
        
        plt.title(f'{metric.capitalize()} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{metric}_over_epochs.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Plot all metrics in a single figure with proper epoch alignment
    plt.figure(figsize=(16, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 2, i)
        for phase in ['train', 'validation']:
            phase_data = train_val_df[train_val_df['phase'] == phase].sort_values('epoch')
            if not phase_data.empty:
                x_values = phase_data['epoch'] if phase == 'train' else phase_data['epoch'] + 0.1  # Slight offset
                plt.plot(x_values, 
                        phase_data[metric], 
                        marker='o' if phase == 'train' else 's',
                        label=f'{phase.capitalize()}',
                        markersize=6,
                        linewidth=1.5)
        
        plt.title(f'{metric.capitalize()}', fontsize=11, pad=8)
        plt.xlabel('Epoch', fontsize=9)
        plt.ylabel(metric.capitalize(), fontsize=9)
        plt.legend(fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis to show integer epochs only
        if not train_val_df.empty:
            min_epoch = int(train_val_df['epoch'].min())
            max_epoch = int(train_val_df['epoch'].max()) + 1
            plt.xticks(range(min_epoch, max_epoch))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/all_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Final metrics comparison
    final_metrics = []
    for phase in ['train', 'validation', 'test']:
        phase_df = df[df['phase'] == phase]
        if not phase_df.empty:
            final_metrics.append({
                'Phase': phase.capitalize(),
                'Accuracy': phase_df['accuracy'].iloc[0],
                'F1 Score': phase_df['f1'].iloc[0],
                'Loss': phase_df['loss'].iloc[0]
            })
    
    if final_metrics:
        final_df = pd.DataFrame(final_metrics)
        plt.figure(figsize=(12, 6))
        
        metrics = ['Accuracy', 'F1 Score', 'Loss']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, i)
            ax = sns.barplot(x='Phase', y=metric, data=final_df)
            plt.title(f'Final {metric}')
            plt.xticks(rotation=45)
            
            # Add value labels
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.4f}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', 
                           xytext=(0, 5), 
                           textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/final_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Add correlation analysis using all epochs
    if not train_val_df.empty:
        # Prepare data for correlation analysis
        metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
        
        # Create separate dataframes for train and validation
        train_df = train_val_df[train_val_df['phase'] == 'train']
        val_df = train_val_df[train_val_df['phase'] == 'validation']
        
        # Create a figure for correlation matrices
        plt.figure(figsize=(16, 6))
        
        # Plot training metrics correlation
        plt.subplot(1, 2, 1)
        train_corr = train_df[metrics].corr()
        mask = np.triu(np.ones_like(train_corr, dtype=bool))
        sns.heatmap(train_corr, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt=".2f",
                   mask=mask,
                   vmin=-1, vmax=1,
                   square=True,
                   cbar_kws={"shrink": .8})
        plt.title('Training Metrics Correlation', fontsize=12, pad=15)
        
        # Plot validation metrics correlation
        plt.subplot(1, 2, 2)
        val_corr = val_df[metrics].corr()
        mask = np.triu(np.ones_like(val_corr, dtype=bool))
        sns.heatmap(val_corr, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt=".2f",
                   mask=mask,
                   vmin=-1, vmax=1,
                   square=True,
                   cbar_kws={"shrink": .8})
        plt.title('Validation Metrics Correlation', fontsize=12, pad=15)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print correlation insights
        print("\nKey Metric Correlations:")
        print("------------------------")
        
        # Loss vs Accuracy
        if len(train_df) > 1:
            train_loss_acc = np.corrcoef(train_df['loss'], train_df['accuracy'])[0,1]
            print(f"Train Loss vs Accuracy: {train_loss_acc:.3f} (strongly negative correlation expected)")
        if len(val_df) > 1:
            val_loss_acc = np.corrcoef(val_df['loss'], val_df['accuracy'])[0,1]
            print(f"Val Loss vs Accuracy: {val_loss_acc:.3f} (strongly negative correlation expected)")
        
        # F1 vs Accuracy
        if len(train_df) > 1:
            train_f1_acc = np.corrcoef(train_df['f1'], train_df['accuracy'])[0,1]
            print(f"\nTrain F1 vs Accuracy: {train_f1_acc:.3f} (high positive correlation expected)")
        if len(val_df) > 1:
            val_f1_acc = np.corrcoef(val_df['f1'], val_df['accuracy'])[0,1]
            print(f"Val F1 vs Accuracy: {val_f1_acc:.3f} (high positive correlation expected)")
        
        # Precision vs Recall
        if len(train_df) > 1:
            train_prec_rec = np.corrcoef(train_df['precision'], train_df['recall'])[0,1]
            print(f"\nTrain Precision vs Recall: {train_prec_rec:.3f} (relationship varies by dataset)")
        if len(val_df) > 1:
            val_prec_rec = np.corrcoef(val_df['precision'], val_df['recall'])[0,1]
            print(f"Val Precision vs Recall: {val_prec_rec:.3f} (relationship varies by dataset)")
    
    print(f"\nAll visualizations saved to: {output_dir}")

if __name__ == "__main__":
    # Default values
    default_metrics_file = "results_bart_20news_centralized/training_metrics_centralized_20250806_111734.csv"
    default_output_dir = "results/plots_centralized"
    
    # Get command line arguments if provided
    if len(sys.argv) > 1:
        metrics_file = sys.argv[1]
    else:
        metrics_file = default_metrics_file
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = default_output_dir
    
    print(f"Visualizing metrics from: {metrics_file}")
    plot_metrics(metrics_file, output_dir)
