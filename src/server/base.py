import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseServer(ABC):
    """Base class for federated learning servers."""
    
    def __init__(self, 
                 args: Any,
                 train_datasets: List[torch.utils.data.Dataset],
                 val_datasets: Optional[List[torch.utils.data.Dataset]] = None,
                 test_datasets: Optional[List[torch.utils.data.Dataset]] = None,
                 **kwargs):
        """Initialize the federated learning server.
        
        Args:
            args: Command line arguments or configuration object
            train_datasets: List of training datasets for each client
            val_datasets: List of validation datasets for each client
            test_datasets: List of test datasets for each client
        """
        self.args = args
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets or [None] * len(train_datasets)
        self.test_datasets = test_datasets or [None] * len(train_datasets)
        
        # Set device
        self.device = torch.device(
            f"cuda:{args.device_id}" if torch.cuda.is_available() and args.device_id >= 0 
            else "cpu"
        )
        
        # Initialize metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
            'bleu': []
        }
        
        # Create output directory
        self.output_dir = getattr(args, 'output_dir', './results')
        os.makedirs(self.output_dir, exist_ok=True)
    
    @abstractmethod
    def select_clients(self, round_idx: int) -> List[int]:
        """Select clients to participate in the current round.
        
        Args:
            round_idx: Current round index
            
        Returns:
            List of selected client indices
        """
        pass
    
    @abstractmethod
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates.
        
        Args:
            client_updates: List of client updates
            
        Returns:
            Aggregated model parameters
        """
        pass
    
    @abstractmethod
    def train_one_round(self, round_idx: int) -> Dict[str, Any]:
        """Train the model for one round of federated learning.
        
        Args:
            round_idx: Current round index
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def evaluate(self, round_idx: int = -1, dataset_type: str = 'val') -> Dict[str, Any]:
        """Evaluate the model on client data.
        
        Args:
            round_idx: Current round index (for logging)
            dataset_type: Type of dataset to evaluate on ('train', 'val', or 'test')
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    @abstractmethod
    def save_model(self, output_dir: str):
        """Save the model to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        pass
    
    @abstractmethod
    def save_metrics(self, output_dir: str):
        """Save metrics to disk.
        
        Args:
            output_dir: Directory to save the metrics
        """
        pass
    
    def _log_metrics(self, metrics: Dict[str, Any], prefix: str = ''):
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            prefix: Prefix to add to metric names
        """
        log_str = f"{prefix} " if prefix else ""
        log_str += ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(log_str)
    
    def _update_metrics(self, metrics: Dict[str, Any], dataset_type: str = 'val'):
        """Update metrics dictionary.
        
        Args:
            metrics: Dictionary of metrics to update
            dataset_type: Type of dataset ('train', 'val', or 'test')
        """
        for k, v in metrics.items():
            if k in self.metrics:
                if k.startswith(dataset_type + '_'):
                    metric_name = k.split('_', 1)[1]
                    self.metrics[metric_name].append(v)
                else:
                    self.metrics[k].append(v)
