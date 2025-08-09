import os
import yaml
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.ticker as mtick

def run_experiment(config_path, num_clients, output_dir, num_rounds=5):
    """Run training with specified number of clients."""
    # Load and update config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config
    config['num_clients'] = num_clients
    config['clients_per_round'] = max(1, num_clients // 2)  # 50% participation rate
    config['num_rounds'] = num_rounds
    
    # Create output directory for this experiment
    exp_output_dir = os.path.join(output_dir, f'clients_{num_clients}')
    os.makedirs(exp_output_dir, exist_ok=True)
    config['output_dir'] = exp_output_dir
    config['model_save_path'] = os.path.join(exp_output_dir, 'saved_models')
    
    # Save updated config
    exp_config_path = os.path.join(exp_output_dir, 'config.yaml')
    with open(exp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Run training
    cmd = f"python train_fed_distilbart_cnndm.py --config {exp_config_path}"
    subprocess.run(cmd, shell=True, check=True)
    
    return exp_output_dir

def collect_results(base_output_dir, num_clients_list):
    """Collect results from all experiments."""
    results = []
    
    for num_clients in num_clients_list:
        exp_dir = os.path.join(base_output_dir, f'clients_{num_clients}')
        
        # Look for metrics files in both root and saved_models directory
        metrics_files = list(Path(exp_dir).rglob('*metrics*.csv'))
        
        if not metrics_files:
            print(f"No metrics files found in {exp_dir}")
            continue
            
        for metrics_file in metrics_files:
            try:
                # Skip client_metrics.csv files - these are not the metrics we're looking for
                if 'client_metrics.csv' in str(metrics_file):
                    continue
                    
                # Read the CSV file with error handling
                df = pd.read_csv(metrics_file, on_bad_lines='warn')
                
                if df.empty:
                    print(f"Empty metrics file: {metrics_file}")
                    continue
                
                # Ensure phase column is string type and handle NaN values
                if 'phase' in df.columns:
                    df['phase'] = df['phase'].astype(str).str.lower()
                
                # Look for evaluation rows
                eval_df = None
                if 'phase' in df.columns:
                    # Try different possible phase values
                    for phase_val in ['eval', 'evaluation', 'valid', 'validation']:
                        phase_mask = df['phase'].str.contains(phase_val, na=False, case=False)
                        if phase_mask.any():
                            eval_df = df[phase_mask]
                            break
                
                # If no evaluation rows found, try to find any row with ROUGE scores
                if eval_df is None or eval_df.empty:
                    print(f"No evaluation rows in {metrics_file}, searching for any row with ROUGE scores")
                    # Look for any row that has ROUGE scores
                    for _, row in df.iterrows():
                        has_rouge = any(f'rouge' in str(col).lower() for col in row.index)
                        if has_rouge:
                            eval_df = pd.DataFrame([row])
                            break
                    
                    # If still no rows found, use the last row
                    if eval_df is None or eval_df.empty:
                        print(f"No rows with ROUGE scores found in {metrics_file}, using last row")
                        eval_df = df.iloc[[-1]]
                
                # Get the last evaluation row
                last_eval = eval_df.iloc[-1]
                
                # Extract metrics, handling different possible column names
                result = {
                    'num_clients': num_clients,
                    'loss': float(last_eval.get('loss', 0))
                }
                
                # Handle ROUGE scores - looking for columns like rouge1, rouge2, rougeL
                rouge_metrics = {}
                
                # First, try to find ROUGE columns in the standard format
                for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                    if rouge_type in last_eval:
                        try:
                            val = float(last_eval[rouge_type])
                            if val > 0:
                                rouge_metrics[rouge_type] = val
                        except (ValueError, TypeError):
                            pass
                
                # Print file path for debugging
                print(f"\nProcessing file: {metrics_file}")
                
                # Read the file as plain text to handle the format
                with open(metrics_file, 'r') as f:
                    lines = f.readlines()
                
                # Find the last evaluation row
                last_eval_line = None
                for line in reversed(lines):
                    if ',eval,' in line:
                        last_eval_line = line.strip()
                        break
                
                if last_eval_line:
                    # Split the CSV line
                    parts = last_eval_line.split(',')
                    
                    # Extract ROUGE scores (they are in positions 5,6,7 after splitting by comma)
                    try:
                        rouge_metrics = {
                            'rouge1': float(parts[5]) * 100,  # Scale to 0-100
                            'rouge2': float(parts[6]) * 100,
                            'rougeL': float(parts[7]) * 100
                        }
                        print(f"Found evaluation row: {rouge_metrics}")
                    except (ValueError, IndexError) as e:
                        print(f"Error extracting ROUGE scores from evaluation row: {e}")
                        rouge_metrics = None
                else:
                    print("No evaluation row found in the file")
                    rouge_metrics = None
                
                # If no evaluation row found, try to find any row with non-zero ROUGE scores
                if not rouge_metrics:
                    for line in reversed(lines):
                        parts = line.strip().split(',')
                        if len(parts) > 7:  # Ensure we have enough columns
                            try:
                                r1, r2, rL = float(parts[5]), float(parts[6]), float(parts[7])
                                if r1 > 0 or r2 > 0 or rL > 0:
                                    rouge_metrics = {
                                        'rouge1': r1 * 100,
                                        'rouge2': r2 * 100,
                                        'rougeL': rL * 100
                                    }
                                    print(f"Using fallback ROUGE scores: {rouge_metrics}")
                                    break
                            except (ValueError, IndexError):
                                continue
                
                # If we found ROUGE scores, add them to results
                if rouge_metrics:
                    result.update(rouge_metrics)
                    
                    results.append(result)
                    print(f"Collected results from {metrics_file}")
                    break  # Process only the first valid metrics file per directory
                else:
                    print(f"No valid ROUGE scores in {metrics_file}")
                    
            except Exception as e:
                print(f"Error processing {metrics_file}: {e}")
                import traceback
                traceback.print_exc()
    
    return pd.DataFrame(results) if results else pd.DataFrame()

def load_contribution_metrics(exp_dir: str) -> Optional[Dict[str, Any]]:
    """Load client contribution metrics from an experiment directory."""
    try:
        # Look for the contribution summary file
        summary_path = os.path.join(exp_dir, 'client_contributions', 'contribution_summary.json')
        if not os.path.exists(summary_path):
            return None
            
        with open(summary_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading contribution metrics from {exp_dir}: {e}")
        return None

def plot_contribution_analysis(metrics_list: List[Dict[str, Any]], output_dir: str) -> None:
    """Generate visualizations for client contributions across experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out None metrics
    metrics_list = [m for m in metrics_list if m is not None]
    if not metrics_list:
        print("No valid contribution metrics found for visualization")
        return
    
    # Prepare data for plotting
    data = []
    for metrics in metrics_list:
        if not metrics:
            continue
            
        num_clients = len(metrics.get('total_contributions', {}))
        if num_clients == 0:
            continue
            
        # Add overall metrics
        data.append({
            'num_clients': num_clients,
            'gini': metrics.get('inequality', {}).get('gini', 0),
            'cv': metrics.get('inequality', {}).get('cv', 0),
            'mean_contribution': metrics.get('inequality', {}).get('mean_contribution', 0),
            'metric': 'Overall'
        })
        
        # Add per-client metrics
        avg_contributions = metrics.get('avg_contributions', {})
        participation = metrics.get('participation', {})
        
        for client_id, contrib in avg_contributions.items():
            data.append({
                'num_clients': num_clients,
                'contribution': contrib,
                'participation': participation.get(client_id, 0),
                'client_id': f'Client {client_id}',
                'metric': 'Per-Client'
            })
    
    if not data:
        print("No contribution data available for visualization")
        return
        
    df = pd.DataFrame(data)
    
    try:
        # Plot 1: Gini Coefficient vs Number of Clients
        plt.figure(figsize=(10, 6))
        overall_df = df[df['metric'] == 'Overall'].drop_duplicates('num_clients')
        if not overall_df.empty:
            sns.lineplot(data=overall_df, x='num_clients', y='gini', 
                         marker='o', label='Gini Coefficient')
            plt.title('Inequality in Client Contributions (Gini Coefficient)')
            plt.xlabel('Number of Clients')
            plt.ylabel('Gini Coefficient')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'gini_vs_clients.png'))
            plt.close()
        
        # Plot 2: Participation Heatmap
        if len(metrics_list) > 1:
            participation_data = []
            for metrics in metrics_list:
                if not metrics:
                    continue
                    
                num_clients = len(metrics.get('total_contributions', {}))
                if num_clients == 0:
                    continue
                    
                # Get the number of clients that participated in each round
                num_participating_clients = {}
                for client_id, part_count in metrics.get('participation', {}).items():
                    num_participating_clients[client_id] = part_count
                
                # Calculate participation rate per client
                total_rounds = metrics.get('total_rounds', 1)
                if total_rounds == 0:
                    total_rounds = 1
                
                # Calculate expected participations (assuming uniform selection)
                # If we have 'num_clients' clients and 'total_rounds' rounds,
                # the expected number of participations per client is total_rounds * (clients_per_round / num_clients)
                # Since we might not know clients_per_round, we'll use the average participation rate
                
                # First pass to get total participations
                total_participations = sum(metrics.get('participation', {}).values())
                if total_participations == 0:
                    total_participations = 1
                
                # Calculate expected participations based on uniform distribution
                num_clients = len(metrics.get('participation', {}))
                if num_clients == 0:
                    num_clients = 1
                
                expected_participations = total_rounds * (total_participations / (num_clients * total_rounds))
                if expected_participations == 0:
                    expected_participations = 1
                
                # Calculate normalized participation percentage
                for client_id, part_count in metrics.get('participation', {}).items():
                    participation_pct = (part_count / expected_participations) * 100
                    # Cap at 100% to avoid values over 100%
                    participation_pct = min(100.0, participation_pct)
                    
                    participation_data.append({
                        'Client': f'Client {client_id}',
                        'Num_Clients': num_clients,
                        'Participation': participation_pct,
                        'Participation_Count': part_count,
                        'Max_Possible': expected_participations
                    })
            
            if participation_data:
                part_df = pd.DataFrame(participation_data)
                pivot_df = part_df.pivot(index='Client', columns='Num_Clients', values='Participation')
                if not pivot_df.empty:
                    plt.figure(figsize=(12, max(6, len(pivot_df) * 0.5)))
                    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='YlGnBu',
                                cbar_kws={'label': 'Participation Rate (%)'})
                    plt.title('Client Participation Heatmap')
                    plt.xlabel('Number of Clients')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'participation_heatmap.png'))
                    plt.close()
        
        # Plot 3: Contribution Distribution
        per_client_df = df[df['metric'] == 'Per-Client']
        if not per_client_df.empty:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=per_client_df, x='num_clients', y='contribution')
            plt.title('Distribution of Client Contributions')
            plt.xlabel('Number of Clients')
            plt.ylabel('Contribution (Normalized)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'contribution_distribution.png'))
            plt.close()
            
    except Exception as e:
        print(f"Error generating contribution visualizations: {e}")
        import traceback
        traceback.print_exc()

def plot_results(results_df, output_dir):
    """Plot comparison of results across different numbers of clients."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to markdown and CSV
    try:
        # Format percentages for better readability
        display_df = results_df.copy()
        for col in ['rouge1', 'rouge2', 'rougeL']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
        
        table = display_df.to_markdown(index=False, floatfmt=".2f")
        
        # Save the table to a markdown file
        table_path = os.path.join(output_dir, 'results_summary.md')
        with open(table_path, 'w') as f:
            f.write("# Federated Learning Results Summary\n\n")
            f.write("## Model Performance by Number of Clients\n\n")
            f.write(table)
            
            # Add interpretation
            f.write("\n\n## Interpretation\n")
            f.write("- **ROUGE Scores**: Higher is better. ROUGE-1 measures unigram overlap, "
                   "ROUGE-2 measures bigram overlap, and ROUGE-L measures longest common subsequence.\n")
            f.write("- **Loss**: Lower is better. Cross-entropy loss on the evaluation set.\n")
            
        print(f"Results summary saved to: {table_path}")
    except Exception as e:
        print(f"Could not generate markdown table: {e}")
        table_path = os.path.join(output_dir, 'results_summary.csv')
        results_df.to_csv(table_path, index=False)
        print(f"Results summary saved to: {table_path}")
    
    # Generate performance plots
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot each ROUGE metric
        for metric, marker, label in [
            ('rouge1', 'o', 'ROUGE-1'),
            ('rouge2', 's', 'ROUGE-2'),
            ('rougeL', 'd', 'ROUGE-L')
        ]:
            if metric in results_df.columns:
                plt.plot(results_df['num_clients'], results_df[metric], 
                        marker=marker, label=label, linewidth=2, markersize=8)
        
        plt.xlabel('Number of Clients', fontsize=12)
        plt.ylabel('ROUGE Score (%)', fontsize=12)
        plt.title('Model Performance vs Number of Clients', fontsize=14, pad=20)
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(results_df['num_clients'].unique())
        
        # Set y-axis limits to better show the ROUGE score range
        plt.ylim(0, max(results_df[['rouge1', 'rouge2', 'rougeL']].max().max() * 1.1, 10))
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plot saved to: {plot_path}")
        return plot_path
        
    except Exception as e:
        print(f"Could not generate performance plot: {e}")
        return None

def collect_contribution_metrics(base_output_dir: str, num_clients_list: List[int]) -> List[Dict[str, Any]]:
    """Collect client contribution metrics from all experiments."""
    metrics_list = []
    for num_clients in num_clients_list:
        exp_dir = os.path.join(base_output_dir, f'clients_{num_clients}')
        metrics = load_contribution_metrics(exp_dir)
        if metrics:
            metrics_list.append(metrics)
    return metrics_list

def main():
    parser = argparse.ArgumentParser(description='Run federated learning experiments with different numbers of clients')
    parser.add_argument('--config', type=str, default='configs/distilbart_cnndm.yaml',
                      help='Path to base configuration YAML file')
    parser.add_argument('--output-dir', type=str, default='./experiment_results',
                      help='Base directory to save experiment results')
    parser.add_argument('--min-clients', type=int, default=2,
                      help='Minimum number of clients to test')
    parser.add_argument('--max-clients', type=int, default=10,
                      help='Maximum number of clients to test')
    parser.add_argument('--num-rounds', type=int, default=5,
                      help='Number of federated rounds to run for each experiment')
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip training and only analyze existing results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments for different numbers of clients
    num_clients_list = list(range(args.min_clients, args.max_clients + 1))
    
    if not args.skip_training:
        print(f"Running experiments for {len(num_clients_list)} different client counts...")
        for num_clients in num_clients_list:
            print(f"\n=== Running experiment with {num_clients} clients for {args.num_rounds} rounds ===")
            run_experiment(args.config, num_clients, args.output_dir, args.num_rounds)
    else:
        print("Skipping training as --skip-training flag is set")
    
    # Collect and plot results
    print("\nCollecting results...")
    
    # Collect model performance metrics
    results_df = collect_results(args.output_dir, num_clients_list)
    
    # Collect client contribution metrics
    print("\nAnalyzing client contributions...")
    metrics_list = collect_contribution_metrics(args.output_dir, num_clients_list)
    
    # Create analysis directory
    analysis_dir = os.path.join(args.output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    if not results_df.empty:
        # Plot performance results
        plot_path = plot_results(results_df, analysis_dir)
        print(f"\nPerformance analysis saved to {os.path.join(analysis_dir, 'results_summary.md')}")
        
        # Save raw results to CSV
        results_df.to_csv(os.path.join(analysis_dir, 'raw_results.csv'), index=False)
    else:
        print("No model performance results found to plot.")
    
    if metrics_list:
        # Plot contribution analysis
        plot_contribution_analysis(metrics_list, analysis_dir)
        print(f"Client contribution analysis saved to {analysis_dir}")
        
        # Save raw metrics to JSON
        with open(os.path.join(analysis_dir, 'contribution_metrics.json'), 'w') as f:
            json.dump(metrics_list, f, indent=2)
    else:
        print("No client contribution metrics found to analyze.")
    
    print(f"\nAnalysis complete. Results saved to: {analysis_dir}")

if __name__ == "__main__":
    main()
