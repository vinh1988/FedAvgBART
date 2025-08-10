import os
import json
import glob
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

def load_metrics_for_clients(base_dir: str, min_clients: int = 2, max_clients: int = 10) -> Dict[int, Dict]:
    """
    Load metrics for different client configurations.
    
    Args:
        base_dir: Base directory containing client experiment results
        min_clients: Minimum number of clients to include
        max_clients: Maximum number of clients to include
        
    Returns:
        Dictionary mapping number of clients to their metrics data
    """
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
                all_metrics[num_clients] = data.get('global_metrics', {})
                print(f"Loaded metrics for {num_clients} clients from {os.path.basename(latest_file)}")
        except Exception as e:
            print(f"Error loading {latest_file}: {e}")
    
    return all_metrics

def extract_metrics(metrics_data: Dict[int, Dict]) -> List[Dict[str, Any]]:
    """
    Extract relevant metrics from the metrics data.
    
    Args:
        metrics_data: Dictionary mapping number of clients to their metrics
        
    Returns:
        List of dictionaries containing extracted metrics
    """
    extracted = []
    
    for num_clients, rounds_data in metrics_data.items():
        for round_num, metrics in rounds_data.items():
            if not isinstance(metrics, dict) or 'phase' not in metrics:
                continue
                
            if metrics['phase'] == 'evaluation':
                # Extract base metrics
                entry = {
                    'num_clients': num_clients,
                    'round': int(round_num),
                    'loss': metrics.get('loss')
                }
                
                # Extract ROUGE metrics (handle both old and new formats)
                for rouge in [1, 2, 'L']:
                    key = f'rouge{rouge}'
                    f1_key = f'rouge{rouge}_f1'
                    
                    if f1_key in metrics:
                        entry[f'rouge{rouge}_f1'] = metrics[f1_key] * 100
                    elif key in metrics:
                        entry[f'rouge{rouge}_f1'] = metrics[key] * 100
                
                # Extract BLEU scores
                for bleu in range(1, 5):
                    key = f'bleu{bleu}'
                    if key in metrics:
                        entry[f'bleu{bleu}'] = metrics[key] * 100
                
                # Extract BERTScore
                if 'bertscore_f1' in metrics:
                    entry['bertscore_f1'] = metrics['bertscore_f1'] * 100
                
                # Extract contribution metrics if available
                contribution_metrics = ['gini_coefficient', 'mean_contribution', 
                                      'min_contribution', 'max_contribution', 'cv_contribution']
                for metric in contribution_metrics:
                    if metric in metrics:
                        entry[metric] = metrics[metric]
                
                extracted.append(entry)
    
    return extracted

def save_metrics_to_csv(metrics: List[Dict[str, Any]], output_path: str):
    """
    Save extracted metrics to a CSV file.
    
    Args:
        metrics: List of metric dictionaries
        output_path: Path to save the CSV file
    """
    if not metrics:
        print("No metrics to save.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(metrics)
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Collect and consolidate metrics from federated learning experiments.'
    )
    parser.add_argument('--base-dir', type=str, default='./experiment_results',
                       help='Base directory containing client experiment results')
    parser.add_argument('--output-file', type=str, 
                       default='./experiment_results/analysis/consolidated_metrics.csv',
                       help='Path to save the consolidated metrics CSV')
    parser.add_argument('--min-clients', type=int, default=2,
                       help='Minimum number of clients to include')
    parser.add_argument('--max-clients', type=int, default=10,
                       help='Maximum number of clients to include')
    
    args = parser.parse_args()
    
    print(f"Collecting metrics for {args.min_clients} to {args.max_clients} clients...")
    
    # Load metrics for all client configurations
    metrics_data = load_metrics_for_clients(
        args.base_dir, 
        min_clients=args.min_clients, 
        max_clients=args.max_clients
    )
    
    if not metrics_data:
        print("No metrics data found for the specified client configurations.")
        return
    
    # Extract and consolidate metrics
    print("Extracting and consolidating metrics...")
    consolidated_metrics = extract_metrics(metrics_data)
    
    # Save to CSV
    save_metrics_to_csv(consolidated_metrics, args.output_file)
    
    # Print summary
    if consolidated_metrics:
        df = pd.DataFrame(consolidated_metrics)
        print("\nSummary of collected metrics:")
        print(f"- Number of client configurations: {df['num_clients'].nunique()}")
        print(f"- Total rounds: {len(df)}")
        print(f"- Available metrics: {', '.join(df.columns)}")
    else:
        print("No metrics were extracted.")

if __name__ == "__main__":
    main()
