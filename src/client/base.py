import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseClient(ABC):
    """Base class for federated learning clients."""
    
    def __init__(self, 
                 client_id: int,
                 train_dataset: Optional[torch.utils.data.Dataset] = None,
                 val_dataset: Optional[torch.utils.data.Dataset] = None,
                 test_dataset: Optional[torch.utils.data.Dataset] = None,
                 device: str = 'cpu',
                 **kwargs):
        """Initialize the federated learning client.
        
        Args:
            client_id: Unique identifier for the client
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.device = device
        
        # Set default batch size if not provided
        self.batch_size = kwargs.get('batch_size', 8)
        self.local_epochs = kwargs.get('local_epochs', 1)
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters.
        
        Returns:
            Dictionary containing model parameters
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters.
        
        Args:
            parameters: Dictionary containing model parameters
        """
        pass
    
    @abstractmethod
    def train(self, num_epochs: int = 1, batch_size: int = 8) -> Dict[str, Any]:
        """Train the model on client data.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def evaluate(self, dataset_type: str = 'test', batch_size: int = 8) -> Dict[str, Any]:
        """Evaluate the model on client data.
        
        Args:
            dataset_type: Type of dataset to evaluate on ('train', 'val', or 'test')
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def get_train_loader(self, batch_size: int = None) -> torch.utils.data.DataLoader:
        """Get training data loader.
        
        Args:
            batch_size: Batch size (defaults to self.batch_size if None)
            
        Returns:
            DataLoader for training data
        """
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")
            
        batch_size = batch_size or self.batch_size
        
        # If dataset is smaller than batch size, reduce batch size
        if len(self.train_dataset) < batch_size:
            batch_size = len(self.train_dataset)
            
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False  # Don't drop last batch to ensure all data is used
        )
    
    def get_val_loader(self, batch_size: int = None) -> torch.utils.data.DataLoader:
        """Get validation data loader.
        
        Args:
            batch_size: Batch size (defaults to self.batch_size if None)
            
        Returns:
            DataLoader for validation data
        """
        if self.val_dataset is None:
            raise ValueError("No validation dataset provided")
            
        batch_size = batch_size or self.batch_size
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
    
    def get_test_loader(self, batch_size: int = None) -> torch.utils.data.DataLoader:
        """Get test data loader.
        
        Args:
            batch_size: Batch size (defaults to self.batch_size if None)
            
        Returns:
            DataLoader for test data
        """
        if self.test_dataset is None:
            raise ValueError("No test dataset provided")
            
        batch_size = batch_size or self.batch_size
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
    
    def save_model(self, output_dir: str):
        """Save the model to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, os.path.join(output_dir, f'model_client_{self.client_id}.pt'))
    
    def load_model(self, model_path: str):
        """Load the model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def _log_metrics(self, metrics: Dict[str, Any], prefix: str = ''):
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            prefix: Prefix to add to metric names
        """
        log_str = f"Client {self.client_id}"
        if prefix:
            log_str += f" - {prefix}"
        log_str += ": " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(log_str)
