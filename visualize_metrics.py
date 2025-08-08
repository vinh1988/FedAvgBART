import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def load_latest_metrics(results_dir='./results'):
    """Load the most recent metrics CSV file."""
    # Find all CSV files in results directory
    csv_files = glob(os.path.join(results_dir, 'training_metrics_*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No metrics files found in {results_dir}")
    
    # Get the most recent file
    latest_file = max(csv_files, key=os.path.getmtime)
    print(f"Loading metrics from: {latest_file}")
    
    # Load and preprocess data
    df = pd.read_csv(latest_file)
    
    # Convert client_id to string for better plotting
    df['client_id'] = df['client_id'].astype(str)
    
    return df, os.path.basename(latest_file)

def plot_metrics(df, output_dir='./results/plots'):
    """Plot training and evaluation metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter evaluation metrics (global model)
    eval_metrics = df[df['client_id'] == 'global']
    
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training and Evaluation Loss
    plt.subplot(2, 2, 1)
    train_metrics = df[df['phase'] == 'train'].groupby('round')['loss'].mean().reset_index()
    sns.lineplot(data=train_metrics, x='round', y='loss', 
                 marker='o', label='Training')
    sns.lineplot(data=eval_metrics, x='round', y='loss', 
                 marker='s', label='Evaluation')
    plt.title('Training vs Evaluation Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot 2: ROUGE-L F1 Score
    plt.subplot(2, 2, 2)
    sns.lineplot(data=eval_metrics, x='round', y='f1', 
                 marker='o', color='green')
    plt.title('ROUGE-L F1 Score')
    plt.xlabel('Round')
    plt.ylabel('F1 Score (%)')
    
    # Plot 3: ROUGE Precision and Recall
    plt.subplot(2, 2, 3)
    sns.lineplot(data=eval_metrics, x='round', y='precision', 
                 marker='o', label='Precision')
    sns.lineplot(data=eval_metrics, x='round', y='recall', 
                 marker='s', label='Recall')
    plt.title('ROUGE-L Precision and Recall')
    plt.xlabel('Round')
    plt.ylabel('Score (%)')
    plt.legend()
    
    # Plot 4: Client Training Loss
    plt.subplot(2, 2, 4)
    train_metrics = df[df['phase'] == 'train']
    sns.lineplot(data=train_metrics, x='round', y='loss', 
                 hue='client_id', style='client_id',
                 markers=True, dashes=False)
    plt.title('Client Training Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend(title='Client ID')
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

if __name__ == "__main__":
    try:
        # Load the latest metrics
        df, filename = load_latest_metrics()
        
        # Print basic statistics
        print("\nMetrics Summary:")
        print("-" * 50)
        print(f"Total Rounds: {df['round'].max()}")
        print(f"Clients: {df[df['phase'] == 'train']['client_id'].unique().tolist()}")
        print("\nFinal Evaluation Metrics:")
        final_metrics = df[df['client_id'] == 'global'].iloc[-1]
        print(f"Loss: {final_metrics['loss']:.4f}")
        print(f"ROUGE-L F1: {final_metrics['f1']:.2f}%")
        print(f"ROUGE-L Precision: {final_metrics['precision']:.2f}%")
        print(f"ROUGE-L Recall: {final_metrics['recall']:.2f}%")
        
        # Generate and save plots
        plot_path = plot_metrics(df)
        print(f"\nPlots saved to: {plot_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
