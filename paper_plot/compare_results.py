import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
BASE_DIR = Path("/mnt/sda1/Projects/jsl/vp_gitlab/FED/FED-OPT-BERT/commit/FED-OPT-BERT-PYTORCH/paper_plot")
OUTPUT_DIR = BASE_DIR / "analysis_results"

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (12, 6),
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def load_results():
    """Load both centralized and federated results."""
    # Load centralized results
    try:
        central_df = pd.read_csv(OUTPUT_DIR / 'centralized_summary.csv')
        central_df['training'] = 'Centralized'
        central_df = central_df.rename(columns={'Model': 'model'})
        # Convert metrics to percentage scale
        for col in ['rouge1', 'rouge2', 'rougeL', 'bleu4']:
            if col in central_df.columns:
                central_df[col] = central_df[col] * 100
    except FileNotFoundError:
        print("Centralized results not found. Run analyze_centralized.py first.")
        central_df = pd.DataFrame()
    
    # Load federated results
    try:
        fed_df = pd.read_csv(OUTPUT_DIR / 'federated_summary.csv')
        fed_df['training'] = 'Federated'
        
        # Handle different column name formats
        column_mapping = {
            'Model': 'model',
            'α': 'alpha',
            'Best Round': 'round',
            'rouge1': 'rouge1',
            'rouge2': 'rouge2',
            'rougeL': 'rougeL',
            'bleu4': 'bleu4',
            'Training Loss': 'train_loss'
        }
        
        # Only include columns that exist in the CSV
        column_mapping = {k: v for k, v in column_mapping.items() 
                         if k in fed_df.columns or k == 'α' and 'alpha' in fed_df.columns}
        
        # Handle special case for alpha (might be 'α' or 'alpha' in the file)
        if 'α' in column_mapping and 'α' not in fed_df.columns and 'alpha' in fed_df.columns:
            column_mapping['alpha'] = 'alpha'
            del column_mapping['α']
            
        fed_df = fed_df.rename(columns=column_mapping)
        
        # Ensure alpha is float and not NaN
        if 'alpha' in fed_df.columns:
            fed_df['alpha'] = fed_df['alpha'].astype(str).str.strip().replace('', '0.5').astype(float)
        else:
            fed_df['alpha'] = 0.5  # Default value if missing
    except FileNotFoundError:
        print("Federated results not found. Run analyze_federated.py first.")
        fed_df = pd.DataFrame()
    
    # Convert metrics to consistent format
    for df in [central_df, fed_df]:
        if not df.empty:
            # Ensure all metric columns exist
            for col in ['rouge1', 'rouge2', 'rougeL', 'bleu4']:
                if col not in df.columns:
                    df[col] = None
            
            # Convert metrics to percentage if needed
            for col in ['rouge1', 'rouge2', 'rougeL', 'bleu4']:
                if col in df.columns and df[col].max() <= 1.0:
                    df[col] = df[col] * 100
    
    # Combine results
    if not central_df.empty and not fed_df.empty:
        # For centralized, add alpha as None if missing
        if 'alpha' not in central_df.columns:
            central_df['alpha'] = None
        
        # For federated, ensure alpha is included
        if 'alpha' not in fed_df.columns:
            fed_df['alpha'] = 0.5  # Default value if missing
        
        # Get common columns
        common_cols = ['model', 'training', 'rouge1', 'rouge2', 'rougeL', 'bleu4', 'train_loss', 'alpha']
        
        # Only include columns that exist in both dataframes
        central_cols = [col for col in common_cols if col in central_df.columns]
        fed_cols = [col for col in common_cols if col in fed_df.columns]
        
        # Combine dataframes
        combined = pd.concat([
            central_df[central_cols],
            fed_df[fed_cols]
        ], ignore_index=True)
        
        return combined
    
    return pd.concat([central_df, fed_df], ignore_index=True)

def save_comparison_data(central_df, fed_df):
    """Save comparison data between centralized and federated results."""
    if central_df is None or fed_df is None:
        print("Missing data for comparison")
        return None
    
    # Make copies to avoid modifying original dataframes
    central_df = central_df.copy()
    fed_df = fed_df.copy()
    
    # Add training type
    central_df['training'] = 'Centralized'
    fed_df['training'] = 'Federated'
    
    # Ensure alpha column exists in federated data
    if 'alpha' not in fed_df.columns:
        fed_df['alpha'] = 0.5  # Default value if missing
    
    # Add alpha to centralized data as None
    central_df['alpha'] = None
    
    # Define columns to include in the output
    output_columns = ['model', 'training', 'rouge1', 'rouge2', 'rougeL', 'bleu4', 'train_loss', 'alpha']
    
    # Filter columns that exist in the dataframes
    central_cols = [col for col in output_columns if col in central_df.columns]
    fed_cols = [col for col in output_columns if col in fed_df.columns]
    
    # Combine data with consistent columns
    combined_df = pd.concat([
        central_df[central_cols],
        fed_df[fed_cols]
    ], ignore_index=True)
    
    # Save combined data
    output_file = OUTPUT_DIR / 'combined_results.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined results to {output_file}")
    
    return combined_df

def generate_comparison_summary(central_df, fed_df):
    """Generate a summary comparison between centralized and federated results."""
    if central_df is None or fed_df is None:
        print("Missing data for comparison summary")
        return None
    
    # Get best results for each model and alpha
    if not central_df.empty:
        central_best = central_df.copy()
        if 'Model' in central_best.columns:
            central_best = central_best.rename(columns={'Model': 'model'})
        central_best['training'] = 'Centralized'
        central_best['alpha'] = None
    else:
        central_best = pd.DataFrame()
    
    if not fed_df.empty:
        # For federated, get best result for each (model, alpha) pair
        if 'model' in fed_df.columns and 'alpha' in fed_df.columns:
            fed_best = fed_df.copy()
            fed_best = fed_best.loc[fed_best.groupby(['model', 'alpha'])['rougeL'].idxmax()]
            fed_best['training'] = 'Federated'
        else:
            fed_best = pd.DataFrame()
    else:
        fed_best = pd.DataFrame()
    
    # Select common metrics
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu4', 'train_loss']
    if not central_best.empty and not fed_best.empty:
        metrics = [m for m in metrics if m in central_best.columns and m in fed_best.columns]
    
    # Prepare results
    result_cols = ['model', 'training', 'alpha'] + metrics
    
    # Combine results
    if not central_best.empty and not fed_best.empty:
        # Select only the columns that exist in both dataframes
        common_cols = [col for col in result_cols if col in central_best.columns and col in fed_best.columns]
        comparison = pd.concat([
            central_best[common_cols],
            fed_best[common_cols]
        ], ignore_index=True)
    elif not central_best.empty:
        common_cols = [col for col in result_cols if col in central_best.columns]
        comparison = central_best[common_cols].copy()
    elif not fed_best.empty:
        common_cols = [col for col in result_cols if col in fed_best.columns]
        comparison = fed_best[common_cols].copy()
    else:
        return None
    
    # Ensure alpha is included in the output
    if 'alpha' not in comparison.columns:
        comparison['alpha'] = None
        
    # Reorder columns
    output_cols = ['model', 'training', 'alpha'] + [col for col in metrics if col in comparison.columns]
    comparison = comparison[output_cols]
    
    # Save comparison
    output_file = OUTPUT_DIR / 'comparison_summary.csv'
    comparison.to_csv(output_file, index=False)
    print(f"Saved comparison summary to {output_file}")
    
    # Create a pivot table for better visualization
    pivot_cols = [col for col in ['rouge1', 'rouge2', 'rougeL', 'bleu4', 'train_loss'] if col in comparison.columns]
    if pivot_cols:
        pivot = comparison.pivot_table(
            index=['model', 'alpha'],
            columns='training',
            values=pivot_cols,
            aggfunc='first'
        )
        # Flatten multi-index columns
        pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
        pivot = pivot.reset_index()
        
        # Save pivot table
        pivot_file = OUTPUT_DIR / 'comparison_pivot.csv'
        pivot.to_csv(pivot_file, index=False)
        print(f"Saved comparison pivot table to {pivot_file}")
        
        # Save as LaTeX table
        latex_file = OUTPUT_DIR / 'comparison_pivot.tex'
        pivot.to_latex(latex_file, index=False, float_format="%.4f")
        print(f"Saved LaTeX table to {latex_file}")
    
    return comparison

def main():
    print("Generating comparison data...")
    
    # Load data
    try:
        central_df = pd.read_csv(OUTPUT_DIR / 'centralized_summary.csv')
        fed_df = pd.read_csv(OUTPUT_DIR / 'federated_summary.csv')
        
        # Save combined data
        combined_df = save_comparison_data(central_df, fed_df)
        
        # Generate comparison summary
        comparison_summary = generate_comparison_summary(central_df, fed_df)
        
        print("\nComparison data generation complete. Results saved to:", OUTPUT_DIR)
        
        return {
            'combined': combined_df,
            'summary': comparison_summary
        }
        
    except FileNotFoundError as e:
        print(f"Error: Required data files not found. {e}")
        return None

if __name__ == "__main__":
    main()
