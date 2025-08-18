import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import os
import logging
import copy
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from server.base import BaseServer
from client.distilbart_client import DistilBARTClient
from models.distilbart_utils import DistilBARTWrapper

logger = logging.getLogger(__name__)

class DistilBARTServer(BaseServer):
    """Server for federated learning with DistilBART model."""
    
    def __init__(self, **kwargs):
        """Initialize DistilBART server.
        
        Args:
            **kwargs: Additional arguments for the base server
        """
        super().__init__(**kwargs)
        
        # Initialize global model
        self.global_model = DistilBARTWrapper(
            model_name=self.args.model_name,
            max_length=self.args.max_source_length,
            max_target_length=self.args.max_target_length,
            device=self.device
        )
        
        # Create clients
        self.clients = self._create_clients()
        
        # Initialize metrics with all required fields
        self.metrics = {
            # Loss metrics
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            
            # ROUGE metrics
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
            
            # BLEU metrics
            'bleu1': [],
            'bleu2': [],
            'bleu3': [],
            'bleu4': [],
            
            # Contribution metrics
            'gini_coefficient': [],
            'mean_contribution': [],
            'min_contribution': [],
            'max_contribution': [],
            'cv_contribution': [],
            'num_samples': []
        }
        
    def select_clients(self, round_idx: int) -> List[int]:
        """Select clients to participate in the current round.
        
        Args:
            round_idx: Current round index
            
        Returns:
            List of selected client indices
        """
        # Simple random selection of clients for this round
        num_clients = min(self.args.clients_per_round, len(self.clients))
        return np.random.choice(
            len(self.clients),
            size=num_clients,
            replace=False
        ).tolist()
        
        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)
    
    def _create_clients(self) -> List[DistilBARTClient]:
        """Create client instances."""
        clients = []
        for client_id in range(self.args.num_clients):
            client = DistilBARTClient(
                client_id=client_id,
                train_dataset=self.train_datasets[client_id] if self.train_datasets else None,
                val_dataset=self.val_datasets[client_id] if self.val_datasets else None,
                test_dataset=self.test_datasets[client_id] if self.test_datasets else None,
                device=self.device,
                args=self.args
            )
            clients.append(client)
        return clients
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using FedAvg.
        
        Args:
            client_updates: List of client updates, where each update is a dictionary
                containing 'model_params' and 'num_samples'
                
        Returns:
            Aggregated model parameters
        """
        # Get total number of samples
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Weighted average of client parameters
        for key in client_updates[0]['model_params'].keys():
            aggregated_params[key] = torch.zeros_like(client_updates[0]['model_params'][key])
            
            for update in client_updates:
                weight = update['num_samples'] / total_samples
                aggregated_params[key] += update['model_params'][key] * weight
        
        return aggregated_params
    
    def train_one_round(self, round_idx: int) -> Dict[str, Any]:
        """Train the model for one round of federated learning.
        
        Args:
            round_idx: Current round index
            
        Returns:
            Dictionary containing training metrics
        """
        # Select client indices for this round
        selected_client_indices = self.select_clients(round_idx)
        
        # Train clients
        client_metrics = []
        client_updates = []
        
        for client_idx in selected_client_indices:
            # Get the client object
            client = self.clients[client_idx]
            
            # Set global model parameters
            client.set_parameters(self.global_model.get_model_parameters())
            
            # Train client
            metrics = client.train(
                num_epochs=self.args.local_epochs,
                batch_size=self.args.batch_size
            )
            
            # Get client update
            client_updates.append({
                'model_params': metrics.pop('model_params'),
                'num_samples': metrics['num_samples']
            })
            
            # Log client metrics
            client_metrics.append(metrics)
            logger.info(f"Round {round_idx} - Client {client.client_id} - Loss: {metrics['loss']:.4f}")
        
        # Aggregate client updates
        aggregated_params = self.aggregate(client_updates)
        
        # Update global model
        self.global_model.set_model_parameters(aggregated_params)
        
        # Log round metrics
        avg_loss = np.mean([m['loss'] for m in client_metrics])
        self.metrics['train_loss'].append(avg_loss)
        
        logger.info(f"Round {round_idx} - Avg Train Loss: {avg_loss:.4f}")
        
        return {
            'round': round_idx,
            'train_loss': avg_loss,
            'num_clients': len(selected_client_indices)
        }
    
    def evaluate(self, round_idx: int = -1, dataset_type: str = 'val') -> Dict[str, Any]:
        """Evaluate the global model on client data.
        
        Args:
            round_idx: Current round index (for logging)
            dataset_type: Type of dataset to evaluate on ('train', 'val', or 'test')
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Set global model to evaluation mode
        self.global_model.model.eval()
        
        # Evaluate on all clients
        client_metrics = []
        
        for client in self.clients:
            # Set global model parameters
            client.set_parameters(self.global_model.get_model_parameters())
            
            # Evaluate client
            metrics = client.evaluate(dataset_type=dataset_type, batch_size=self.args.batch_size)
            client_metrics.append(metrics)
            
            # Log client metrics with all BLEU scores
            logger.info(
                f"Round {round_idx} - Client {client.client_id} - {dataset_type.capitalize()} - "
                f"Loss: {metrics['loss']:.4f}, ROUGE-1: {metrics['rouge1']:.4f}, "
                f"ROUGE-2: {metrics['rouge2']:.4f}, ROUGE-L: {metrics['rougeL']:.4f}, "
                f"BLEU-1: {metrics['bleu1']:.4f}, BLEU-4: {metrics['bleu4']:.4f}"
            )
        
        # Calculate average metrics with robust handling for missing metrics
        avg_metrics = {
            'loss': np.mean([m.get('loss', 0.0) for m in client_metrics]),
            'rouge1': np.mean([m.get('rouge1', 0.0) for m in client_metrics]),
            'rouge2': np.mean([m.get('rouge2', 0.0) for m in client_metrics]),
            'rougeL': np.mean([m.get('rougeL', 0.0) for m in client_metrics]),
            'bleu1': np.mean([m.get('bleu1', 0.0) for m in client_metrics]),
            'bleu2': np.mean([m.get('bleu2', 0.0) for m in client_metrics]),
            'bleu3': np.mean([m.get('bleu3', 0.0) for m in client_metrics]),
            'bleu4': np.mean([m.get('bleu4', 0.0) for m in client_metrics]),
            'num_samples': sum(m.get('num_samples', 1) for m in client_metrics),  # Default to 1 if missing
            'gini_coefficient': np.mean([m.get('gini_coefficient', 0.0) for m in client_metrics]),
            'mean_contribution': np.mean([m.get('mean_contribution', 0.0) for m in client_metrics]),
            'min_contribution': np.min([m.get('min_contribution', 0.0) for m in client_metrics]),
            'max_contribution': np.max([m.get('max_contribution', 0.0) for m in client_metrics]),
            'cv_contribution': np.mean([m.get('cv_contribution', 0.0) for m in client_metrics])
        }
        
        # Log average metrics with BLEU-1 and BLEU-4 scores
        logger.info(
            f"Round {round_idx} - {dataset_type.capitalize()} - "
            f"Avg Loss: {avg_metrics['loss']:.4f}, "
            f"ROUGE-1: {avg_metrics['rouge1']:.4f}, "
            f"ROUGE-2: {avg_metrics['rouge2']:.4f}, "
            f"ROUGE-L: {avg_metrics['rougeL']:.4f}, "
            f"BLEU-1: {avg_metrics['bleu1']:.4f}, "
            f"BLEU-4: {avg_metrics['bleu4']:.4f}"
        )
        
        # Update metrics
        if dataset_type == 'val':
            self.metrics['val_loss'].append(avg_metrics['loss'])
            self.metrics['rouge1'].append(avg_metrics['rouge1'])
            self.metrics['rouge2'].append(avg_metrics['rouge2'])
            self.metrics['rougeL'].append(avg_metrics['rougeL'])
            # Update BLEU metrics
            self.metrics['bleu1'].append(avg_metrics['bleu1'])
            self.metrics['bleu2'].append(avg_metrics.get('bleu2', 0.0))  # Use 0.0 if not present
            self.metrics['bleu3'].append(avg_metrics.get('bleu3', 0.0))  # Use 0.0 if not present
            self.metrics['bleu4'].append(avg_metrics['bleu4'])
            # Update contribution-related metrics to avoid missing values in CSV
            self.metrics['gini_coefficient'].append(avg_metrics.get('gini_coefficient', 0.0))
            self.metrics['mean_contribution'].append(avg_metrics.get('mean_contribution', 0.0))
            self.metrics['min_contribution'].append(avg_metrics.get('min_contribution', 0.0))
            self.metrics['max_contribution'].append(avg_metrics.get('max_contribution', 0.0))
            self.metrics['cv_contribution'].append(avg_metrics.get('cv_contribution', 0.0))
            self.metrics['num_samples'].append(avg_metrics.get('num_samples', 0))
        elif dataset_type == 'test':
            self.metrics['test_loss'].append(avg_metrics['loss'])
        
        return avg_metrics
    
    def save_model(self, output_dir: str):
        """Save model to directory."""
        self.global_model.save_pretrained(output_dir)
    
    def save_metrics(self, output_dir: str):
        """Save metrics to file in both JSON and CSV formats."""
        import json
        import csv
        import numpy as np
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure all metrics are lists and have the same length
        max_len = max(len(v) for v in self.metrics.values() if isinstance(v, list))

        # Special alignment: place test_loss at the final round position
        # since we only evaluate test once at the end of training.
        if 'test_loss' in self.metrics and isinstance(self.metrics['test_loss'], list):
            test_vals = [v for v in self.metrics['test_loss'] if v is not None]
            if test_vals and len(self.metrics['test_loss']) < max_len:
                aligned = [None] * max_len
                # Put the last observed non-None test loss at the last round index
                aligned[-1] = test_vals[-1]
                self.metrics['test_loss'] = aligned

        # Pad remaining metric lists to max length
        for key in self.metrics:
            if isinstance(self.metrics[key], list) and len(self.metrics[key]) < max_len:
                self.metrics[key].extend([None] * (max_len - len(self.metrics[key])))
        
        # Save metrics in JSON format
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2, default=float)
        
        # Define CSV fieldnames with all metrics
        fieldnames = [
            'round', 
            'train_loss', 'val_loss', 'test_loss',  # Loss metrics
            'rouge1', 'rouge2', 'rougeL',         # ROUGE metrics
            'bleu1', 'bleu2', 'bleu3', 'bleu4',   # BLEU metrics
            'gini_coefficient', 'mean_contribution', 'min_contribution', 
            'max_contribution', 'cv_contribution', 'num_samples'  # Contribution metrics
        ]
        
        # Prepare data for CSV
        csv_data = []
        num_rounds = max_len
        
        for round_idx in range(num_rounds):
            row = {'round': round_idx + 1}
            
            # Add all metrics to the row
            for metric in fieldnames[1:]:  # Skip 'round' field
                if metric in self.metrics and len(self.metrics[metric]) > round_idx:
                    value = self.metrics[metric][round_idx]
                    # Convert numpy types to Python native types for JSON serialization
                    if isinstance(value, (np.generic, np.ndarray)):
                        value = value.item() if value.size == 1 else value.tolist()
                    row[metric] = value
                else:
                    row[metric] = None
            
            csv_data.append(row)
        
        # Write CSV file
        csv_path = output_dir / 'metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_data:
                # Ensure all values are serializable
                serializable_row = {}
                for k, v in row.items():
                    if isinstance(v, (np.generic, np.ndarray)):
                        serializable_row[k] = v.item() if hasattr(v, 'item') else float(v)
                    elif v is None or isinstance(v, (int, float, str, bool)):
                        serializable_row[k] = v
                    else:
                        serializable_row[k] = str(v)
                writer.writerow(serializable_row)
        
        # Save best model metrics based on ROUGE-L score
        if 'rougeL' in self.metrics and len(self.metrics['rougeL']) > 0:
            try:
                # Find the best round based on ROUGE-L
                valid_rouge = [x for x in self.metrics['rougeL'] if x is not None]
                if valid_rouge:
                    best_round = int(np.argmax(valid_rouge))
                    best_metrics = {
                        'best_round': best_round + 1,  # 1-based indexing
                        'best_rouge1': float(self.metrics['rouge1'][best_round]) if 'rouge1' in self.metrics and len(self.metrics['rouge1']) > best_round else 0.0,
                        'best_rouge2': float(self.metrics['rouge2'][best_round]) if 'rouge2' in self.metrics and len(self.metrics['rouge2']) > best_round else 0.0,
                        'best_rougeL': float(self.metrics['rougeL'][best_round]),
                        'best_bleu1': float(self.metrics['bleu1'][best_round]) if 'bleu1' in self.metrics and len(self.metrics['bleu1']) > best_round else 0.0,
                        'best_bleu4': float(self.metrics['bleu4'][best_round]) if 'bleu4' in self.metrics and len(self.metrics['bleu4']) > best_round else 0.0,
                        'best_train_loss': float(self.metrics['train_loss'][best_round]) if 'train_loss' in self.metrics and len(self.metrics['train_loss']) > best_round else float('inf'),
                        'best_val_loss': float(self.metrics['val_loss'][best_round]) if 'val_loss' in self.metrics and len(self.metrics['val_loss']) > best_round else float('inf')
                    }
                    
                    # Add contribution metrics for the best round
                    for metric in ['gini_coefficient', 'mean_contribution', 'min_contribution', 'max_contribution', 'cv_contribution', 'num_samples']:
                        if metric in self.metrics and len(self.metrics[metric]) > best_round and self.metrics[metric][best_round] is not None:
                            best_metrics[f'best_{metric}'] = float(self.metrics[metric][best_round])
                    
                    with open(output_dir / 'best_metrics.json', 'w') as f:
                        json.dump(best_metrics, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving best metrics: {e}", exc_info=True)
