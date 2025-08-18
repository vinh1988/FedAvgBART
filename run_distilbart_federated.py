#!/usr/bin/env python3

import os
import sys
import yaml
import torch
import argparse
import logging
from datetime import datetime
from pathlib import Path

from main import main as federated_main
from src import set_logger, set_seed

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
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_output_dir(config, args):
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or config.get('output_dir', 'results_distilbart_cnndm_federated'))
    output_dir = output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return output_dir

def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.num_clients is not None:
        config['num_clients'] = args.num_clients
    if args.num_rounds is not None:
        config['num_rounds'] = args.num_rounds
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Set up output directory
    output_dir = setup_output_dir(config, args)
    
    # Set up logging
    log_file = output_dir / 'training.log'
    logger = set_logger('distilbart_federated', log_file=log_file)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Log config
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Set up TensorBoard
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
    
    try:
        # Convert config to argparse.Namespace for compatibility
        from argparse import Namespace
        args = Namespace(**config)
        
        # Set output_dir in args
        args.output_dir = str(output_dir)
        
        # Run federated learning
        federated_main(args, writer)
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        writer.close()
        logger.info("Training completed!")

if __name__ == "__main__":
    import traceback
    main()
