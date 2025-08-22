#!/usr/bin/env python3

import os
import sys
import yaml
import logging
import torch
import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import csv

# Add the src directory to the path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('distilbart_experiment.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    from src.server.distilbart_server import DistilBARTServer
    from src.client.distilbart_client import DistilBARTClient
    from src.utils import set_seed, TqdmToLogger
    
    # Import dataset and data utilities
    try:
        from src.datasets.cnndm import CNNDMDataset, load_cnndm
        logger.info("Successfully imported CNNDMDataset and load_cnndm from src.datasets.cnndm")
    except ImportError as e:
        logger.error(f"Failed to import required dataset modules: {e}")
        raise
    
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error(f"Python path: {sys.path}")
    raise

def parse_args():
    parser = argparse.ArgumentParser(description="Run Federated Learning with DistilBART on CNN/DM")
    parser.add_argument("--config", type=str, default="configs/distilbart_cnndm_federated.yaml",
                        help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--num_clients", type=int, default=None,
                        help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=None,
                        help="Number of federated rounds")
    parser.add_argument("--local_epochs", type=int, default=None,
                        help="Number of local epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--dirichlet_alpha", type=float, default=None,
                        help="Dirichlet alpha for non-IID client splits (None for IID)")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_output_dir(config, args):
    # Create output directory grouped by num_clients with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.output_dir or config.get('output_dir', 'results_distilbart_cnndm_federated'))
    nc = config.get('num_clients', 'nc')
    output_dir = base_dir / f"nc_{nc}" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return output_dir

def save_data_distribution(output_dir: Path,
                           train_datasets: List[Any],
                           val_datasets: List[Any],
                           test_datasets: List[Any],
                           num_clients: int) -> None:
    """Save per-client dataset sizes and percentage contribution.

    Creates:
    - client_contributions/client_metrics.csv with columns:
      client_id,train_samples,val_samples,test_samples,contribution,percentage,num_clients
    - client_contributions/summary.json with totals.
    """
    contrib_dir = output_dir / 'client_contributions'
    contrib_dir.mkdir(parents=True, exist_ok=True)

    def ds_len(ds):
        try:
            return len(ds) if ds is not None else 0
        except Exception:
            return 0

    train_sizes = [ds_len(ds) for ds in train_datasets]
    val_sizes = [ds_len(ds) for ds in val_datasets]
    test_sizes = [ds_len(ds) for ds in test_datasets]
    total_train = sum(train_sizes)

    csv_path = contrib_dir / 'client_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['client_id','train_samples','val_samples','test_samples','contribution','percentage','num_clients'])
        for cid in range(num_clients):
            tr = train_sizes[cid] if cid < len(train_sizes) else 0
            va = val_sizes[cid] if cid < len(val_sizes) else 0
            te = test_sizes[cid] if cid < len(test_sizes) else 0
            contrib = tr
            pct = (contrib / total_train) if total_train > 0 else 0.0
            writer.writerow([cid, tr, va, te, contrib, pct, num_clients])

    summary = {
        'num_clients': num_clients,
        'total_train_samples': total_train,
        'total_val_samples': sum(val_sizes),
        'total_test_samples': sum(test_sizes),
        'train_samples_per_client': train_sizes,
        'val_samples_per_client': val_sizes,
        'test_samples_per_client': test_sizes,
    }
    with open(contrib_dir / 'summary.json', 'w') as jf:
        json.dump(summary, jf, indent=2)
    logger.info(f"Saved data distribution to {csv_path}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.num_clients is not None:
        config['num_clients'] = args.num_clients
    if args.num_rounds is not None:
        config['num_rounds'] = args.num_rounds
    if args.local_epochs is not None:
        config['local_epochs'] = args.local_epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.seed is not None:
        config['seed'] = args.seed
    if args.dirichlet_alpha is not None:
        config['dirichlet_alpha'] = args.dirichlet_alpha
    
    # Set up output directory
    output_dir = setup_output_dir(config, args)
    
    # Log file is already set up in the global logging configuration
    # All logging will be written to 'distilbart_experiment.log' in the current directory
    
    # Set random seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Log config
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Load dataset
    logger.info("Loading CNN/DM dataset...")
    train_datasets, val_datasets, test_datasets, _ = load_cnndm(
        data_dir=config['data_dir'],
        num_clients=config['num_clients'],
        test_size=0.1,
        random_state=config['seed'],
        max_train_samples=config.get('max_train_samples'),
        max_val_samples=config.get('max_val_samples'),
        max_test_samples=config.get('max_test_samples'),
        dirichlet_alpha=config.get('dirichlet_alpha')
    )
    
    # Initialize server
    logger.info("Initializing server...")
    from types import SimpleNamespace
    # Add device_id to config
    config['device_id'] = config.get('device_id', 0)  # Default to device 0 if not specified
    args = SimpleNamespace(**config)
    server = DistilBARTServer(
        args=args,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        test_datasets=test_datasets
    )

    # Save data distribution snapshot for this run
    try:
        save_data_distribution(output_dir, train_datasets, val_datasets, test_datasets, config['num_clients'])
    except Exception as e:
        logger.warning(f"Failed to save data distribution: {e}")

    # Training loop
    logger.info("Starting federated training...")
    import time
    times = {
        'round_times_sec': [],
        'total_time_sec': None,
        'num_rounds': config['num_rounds'],
        'num_clients': config['num_clients'],
    }
    total_start = time.time()
    for round_idx in range(1, config['num_rounds'] + 1):
        round_start = time.time()
        # Train for one round
        train_metrics = server.train_one_round(round_idx)
        
        # Evaluate on validation set
        if round_idx % config.get('eval_every', 1) == 0 or round_idx == config['num_rounds']:
            val_metrics = server.evaluate(round_idx, dataset_type='val')
            
            # Save model checkpoints
            if config.get('save_model', False) and round_idx % config.get('save_model_freq', 1) == 0:
                model_dir = output_dir / f"round_{round_idx}"
                server.save_model(str(model_dir))
        times['round_times_sec'].append(time.time() - round_start)
    times['total_time_sec'] = time.time() - total_start
    
    # Final evaluation on test set
    logger.info("Final evaluation on test set...")
    test_metrics = server.evaluate(config['num_rounds'], dataset_type='test')
    
    # Save final model and metrics
    logger.info("Saving final model and metrics...")
    server.save_model(str(output_dir / 'final_model'))
    server.save_metrics(str(output_dir))
    # Save efficiency metrics
    try:
        with open(output_dir / 'efficiency.json', 'w') as ef:
            json.dump(times, ef, indent=2)
        logger.info(f"Saved efficiency metrics to {output_dir / 'efficiency.json'}")
    except Exception as e:
        logger.warning(f"Failed to save efficiency metrics: {e}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
