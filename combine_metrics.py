import os
import pandas as pd
from pathlib import Path

def combine_metrics():
    # Directory containing the CSV files
    results_dir = Path("results")
    
    # List all CSV files in the results directory
    csv_files = list(results_dir.glob("training_metrics_*.csv"))
    
    if not csv_files:
        print("No CSV files found in the results directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files to combine.")
    
    # Read and combine all CSV files
    all_dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Add a column to track the source file
            df['source_file'] = file.name
            all_dfs.append(df)
            print(f"Processed {file.name} with {len(df)} rows")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not all_dfs:
        print("No valid CSV files to combine.")
        return
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by round, client_id, and epoch
    if 'round' in combined_df.columns and 'client_id' in combined_df.columns and 'epoch' in combined_df.columns:
        combined_df = combined_df.sort_values(by=['round', 'client_id', 'epoch'])
    
    # Save the combined DataFrame to a new CSV file
    output_file = results_dir / "combined_training_metrics.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nCombined metrics saved to: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print("\nColumn names:", combined_df.columns.tolist())
    print("\nSample data:")
    print(combined_df.head())

if __name__ == "__main__":
    combine_metrics()
