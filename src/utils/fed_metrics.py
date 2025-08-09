import numpy as np
import os
import json
import csv
import torch
from collections import defaultdict

def gini_coefficient(x) -> float:
    """
    Calculate the Gini coefficient of an array or list.
    
    Args:
        x: Input array or list of numerical values
        
    Returns:
        float: Gini coefficient (0-1)
    """
    try:
        # Convert input to numpy array if it's not already
        x = np.asarray(x, dtype=np.float64)
        
        # Remove zeros and sort
        x = np.sort(x[x > 0])
        n = len(x)
        
        if n == 0:
            return 0.0
            
        if np.ptp(x) == 0:  # All values are the same
            return 0.0
            
        # Calculate Gini coefficient using the relative mean difference method
        index = np.arange(1, n + 1)
        gini = np.sum((2 * index - n - 1) * x) / (n * np.sum(x))
        return float(gini)
        
    except Exception as e:
        print(f"Error calculating Gini coefficient: {e}")
        return 0.0

class ClientContributionTracker:
    def __init__(self, num_clients, output_dir):
        self.num_clients = num_clients
        self.output_dir = output_dir
        self.round_metrics = []
        self.client_contributions = defaultdict(list)
        os.makedirs(output_dir, exist_ok=True)
        
    @staticmethod
    def calculate_model_update(global_weights, local_weights):
        """Calculate model update as difference between local and global weights."""
        update = {}
        for key in global_weights:
            update[key] = local_weights[key] - global_weights[key]
        return update

    @staticmethod
    def vectorize_updates(updates):
        """Convert dictionary of updates to a vector."""
        return torch.cat([update.view(-1) for update in updates.values()])

    def calculate_contribution_metrics(self, client_updates, client_sizes, client_ids):
        """Calculate various contribution metrics for the round."""
        if not client_updates:
            return {}

        # Calculate contribution as L2 norm of updates
        contributions = [torch.norm(self.vectorize_updates(update)).item() 
                        for update in client_updates]
        
        # Calculate metrics
        metrics = {
            'contributions': dict(zip(client_ids, contributions)),
            'gini': float(gini_coefficient(contributions)),
            'mean_contribution': float(np.mean(contributions)),
            'std_contribution': float(np.std(contributions)),
            'min_contribution': float(np.min(contributions)),
            'max_contribution': float(np.max(contributions)),
            'cv_contribution': float(np.std(contributions) / np.mean(contributions)) if np.mean(contributions) > 0 else 0,
            'client_sizes': dict(zip(client_ids, client_sizes)),
            'num_participants': len(client_ids)
        }
        
        # Min-max scale contributions
        min_contrib = metrics['min_contribution']
        max_contrib = metrics['max_contribution']
        if max_contrib > min_contrib:
            metrics['scaled_contributions'] = {
                cid: (contrib - min_contrib) / (max_contrib - min_contrib)
                for cid, contrib in zip(client_ids, contributions)
            }
        else:
            metrics['scaled_contributions'] = {cid: 1.0 for cid in client_ids}
            
        return metrics

    def log_round(self, round_num, client_updates, client_sizes, client_ids):
        """Log metrics for a round of federated learning."""
        metrics = self.calculate_contribution_metrics(client_updates, client_sizes, client_ids)
        if not metrics:
            return None

        # Store round metrics
        round_metrics = {
            'round': round_num,
            **{k: v for k, v in metrics.items() if not isinstance(v, dict)}
        }
        self.round_metrics.append(round_metrics)

        # Update client histories
        for i, client_id in enumerate(client_ids):
            self.client_contributions[client_id].append({
                'round': round_num,
                'contribution': metrics['contributions'][client_id],
                'scaled_contribution': metrics['scaled_contributions'][client_id],
                'data_size': client_sizes[i]
            })

        # Save metrics periodically
        if round_num % 5 == 0:  # Save every 5 rounds
            self.save_metrics()

        return metrics

    def save_metrics(self):
        """Save all metrics to files."""
        # Save round metrics
        round_file = os.path.join(self.output_dir, 'round_metrics.json')
        with open(round_file, 'w') as f:
            json.dump(self.round_metrics, f, indent=2)

        # Save client metrics
        client_file = os.path.join(self.output_dir, 'client_metrics.csv')
        with open(client_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['client_id', 'round', 'contribution', 'scaled_contribution', 'data_size'])
            for client_id, metrics in self.client_contributions.items():
                for m in metrics:
                    writer.writerow([
                        client_id,
                        m['round'],
                        m['contribution'],
                        m['scaled_contribution'],
                        m['data_size']
                    ])

    def generate_summary(self) -> dict:
        """
        Generate a summary of client contributions and participation.
        
        Returns:
            dict: Summary containing metrics and statistics
        """
        # Calculate total contributions per client
        total_contributions = {}
        for client_id in range(self.num_clients):
            # Sum up all contributions for this client across rounds
            client_contribs = [m['contribution'] for m in self.client_contributions[client_id]]
            total_contributions[int(client_id)] = float(sum(client_contribs)) if client_contribs else 0.0
            
        # Calculate participation counts
        participation = {int(client_id): int(len(contribs)) 
                        for client_id, contribs in self.client_contributions.items()}
        
        # Calculate inequality metrics on total contributions
        contributions = np.array(list(total_contributions.values()))
        if len(contributions) > 0 and np.any(contributions > 0):
            gini = float(gini_coefficient(contributions))
            cv = float(np.std(contributions) / np.mean(contributions)) if np.mean(contributions) > 0 else 0.0
            mean_contrib = float(np.mean(contributions))
            min_contrib = float(np.min(contributions))
            max_contrib = float(np.max(contributions))
        else:
            gini = 0.0
            cv = 0.0
            mean_contrib = 0.0
            min_contrib = 0.0
            max_contrib = 0.0
            
        # Calculate average contributions per participation
        avg_contributions = {}
        for client_id in total_contributions:
            part_count = participation.get(client_id, 1)  # Avoid division by zero
            avg_contributions[client_id] = total_contributions[client_id] / part_count if part_count > 0 else 0.0
            
        summary = {
            'total_contributions': total_contributions,
            'avg_contributions': avg_contributions,
            'participation': participation,
            'inequality': {
                'gini': gini,
                'cv': cv,
                'mean_contribution': mean_contrib,
                'min_contribution': min_contrib,
                'max_contribution': max_contrib,
            },
            'total_rounds': len(self.round_metrics)
        }
        
        # Save summary to file
        os.makedirs(self.output_dir, exist_ok=True)
        summary_file = os.path.join(self.output_dir, 'contribution_summary.json')
        
        # Convert all numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(x) for x in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
            
        serializable_summary = convert_to_serializable(summary)
        
        with open(summary_file, 'w') as f:
            json.dump(serializable_summary, f, indent=2)
            
        return serializable_summary
