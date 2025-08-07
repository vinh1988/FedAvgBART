import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from rouge_score import rouge_scorer
from datetime import datetime
import yaml
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.distilbart_gen import DistilBARTGen
from src.datasets.cnndm import load_cnndm

# Initialize ROUGE scorer
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def load_config(config_path=None):
    """Load configuration from YAML file with defaults."""
    # Default configuration with explicit types
    default_config = {
        # Experiment settings
        'experiment_name': 'fed_distilbart_cnndm',
        'num_rounds': 13,
        'seed': 42,
        
        # Federated learning settings
        'num_clients': 10,
        'clients_per_round': 2,
        'local_epochs': 1,
        'batch_size': 8,
        'learning_rate': 5e-5,  # float
        'weight_decay': 0.01,   # float
        'warmup_steps': 500,    # int
        'max_grad_norm': 1.0,   # float
        
        # Model settings
        'model_name': 'facebook/bart-large-cnn',
        'max_source_length': 1024,  # int
        'max_target_length': 142,   # int
        
        # Data settings
        'data_dir': './data/cnndm',
        'train_split': 'train',
        'val_split': 'validation',
        'test_split': 'test',
        'max_train_samples': None,  # Set to None to use full dataset
        'max_val_samples': None,    # Set to None to use full dataset
        'max_test_samples': None,   # Set to None to use full dataset
        
        # Output settings
        'output_dir': './results/fed_distilbart_cnndm',
        'model_save_path': './saved_models/fed_distilbart_cnndm',
        'logging_steps': 100,  # int
        'eval_steps': 500,     # int
        'save_steps': 1000,    # int
        'use_wandb': False,    # bool
        
        # Generation parameters
        'num_beams': 4,               # int
        'length_penalty': 2.0,        # float
        'no_repeat_ngram_size': 3,    # int
        'num_workers': 2              # int
    }
    
    # If no config file is provided, return defaults
    if config_path is None or not os.path.exists(config_path):
        print(f"Using default configuration (config file not found at {config_path})")
        return default_config
    
    # Load YAML config
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f) or {}  # Ensure we get a dict even if file is empty
    
    # Type conversion mapping
    type_map = {
        # int fields
        'num_rounds': int, 'seed': int, 'num_clients': int, 'clients_per_round': int,
        'local_epochs': int, 'batch_size': int, 'warmup_steps': int, 'max_source_length': int,
        'max_target_length': int, 'max_train_samples': int, 'max_val_samples': int,
        'max_test_samples': int, 'logging_steps': int, 'eval_steps': int, 'save_steps': int,
        'num_beams': int, 'no_repeat_ngram_size': int, 'num_workers': int,
        
        # float fields
        'learning_rate': float, 'weight_decay': float, 'max_grad_norm': float,
        'length_penalty': float,
        
        # bool fields
        'use_wandb': lambda x: str(x).lower() in ('true', '1', 't', 'y', 'yes')
    }
    
    # Update defaults with YAML config, applying type conversion
    for key, value in yaml_config.items():
        if key in type_map and value is not None:
            try:
                default_config[key] = type_map[key](value)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert {key}={value} to {type_map[key].__name__}. Using default: {default_config[key]}")
        else:
            default_config[key] = value
    
    # Ensure necessary paths exist
    os.makedirs(default_config['output_dir'], exist_ok=True)
    os.makedirs(default_config['model_save_path'], exist_ok=True)
    
    # Set random seed
    np.random.seed(default_config['seed'])
    torch.manual_seed(default_config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(default_config['seed'])
    
    return default_config

def calculate_rouge_metrics(predictions, references):
    """Calculate ROUGE metrics for a list of predictions and references."""
    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores for each prediction-reference pair
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        if not isinstance(pred, str):
            pred = ' '.join(pred) if isinstance(pred, list) else str(pred)
        if not isinstance(ref, str):
            ref = ' '.join(ref) if isinstance(ref, list) else str(ref)
        
        if pred.strip() and ref.strip():  # Only score non-empty strings
            scores = rouge.score(ref, pred)
            rouge_scores.append(scores['rougeL'])
    
    if not rouge_scores:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Calculate average scores
    avg_precision = sum(s.precision for s in rouge_scores) / len(rouge_scores)
    avg_recall = sum(s.recall for s in rouge_scores) / len(rouge_scores)
    avg_f1 = sum(s.fmeasure for s in rouge_scores) / len(rouge_scores)
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1
    }

def init_metrics_logger(output_dir, experiment_name):
    """
    Initialize CSV file for logging federated training metrics.
    
    Args:
        output_dir (str): Directory to save the metrics file
        experiment_name (str): Name of the experiment for the metrics file
        
    Returns:
        str: Path to the created metrics file
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(output_dir, f"{experiment_name}_metrics_{timestamp}.csv")
    
    # Write header with additional metadata
    with open(metrics_file, 'w') as f:
        f.write("# Federated DistilBART Training Metrics - CNN/DailyMail\n")
        f.write("# Generated at: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("# Model: distilbart-cnn-12-6\n")
        f.write("# Task: Text Generation (Summarization)\n")
        f.write("round,client_id,epoch,phase,loss,accuracy,precision,recall,f1\n")
    
    print(f"Metrics will be logged to: {metrics_file}")
    return metrics_file

def log_metrics(metrics_file, metrics):
    """Log metrics to CSV file."""
    with open(metrics_file, 'a') as f:
        f.write(','.join([
            str(metrics.get('round', '')),
            str(metrics.get('client_id', '')),
            str(metrics.get('epoch', '')),
            str(metrics.get('phase', '')),
            f"{metrics.get('loss', 0.0):.4f}",
            f"{metrics.get('accuracy', 0.0):.4f}",
            f"{metrics.get('precision', 0.0):.4f}",
            f"{metrics.get('recall', 0.0):.4f}",
            f"{metrics.get('f1', 0.0):.4f}"
        ]) + '\n')

def train(config_path=None):
    # Load configuration
    config = load_config(config_path)
    
    # Print configuration
    print("\n" + "="*50)
    print("Training Configuration:")
    print("="*50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("="*50 + "\n")
    
    # Initialize metrics logging
    metrics_file = init_metrics_logger(
        output_dir=config['output_dir'],
        experiment_name=config['experiment_name']
    )
    print(f"Model checkpoints will be saved to: {config['model_save_path']}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load full dataset
    print("Loading full CNN/DailyMail dataset...")
    train_datasets, test_datasets, tokenizer = load_cnndm(
        data_dir=config['data_dir'], 
        num_clients=config['num_clients'], 
        test_size=0.1, 
        random_state=config['seed'],
        max_train_samples=config['max_train_samples'] * config['num_clients'] if config['max_train_samples'] else None,
        max_test_samples=config['max_test_samples'] * config['num_clients'] if config['max_test_samples'] else None
    )
    
    # Print dataset statistics
    total_train_samples = sum(len(ds) for ds in train_datasets)
    total_test_samples = sum(len(ds) for ds in test_datasets)
    print(f"\nDataset Statistics:")
    print(f"- Number of clients: {len(train_datasets)}")
    print(f"- Training samples per client: {[len(ds) for ds in train_datasets]}")
    print(f"- Test samples per client: {[len(ds) for ds in test_datasets]}")
    print(f"- Total training samples: {total_train_samples}")
    print(f"- Total test samples: {total_test_samples}")
    
    # Initialize global model
    print("Initializing DistilBART model...")
    model = DistilBARTGen(use_pt_model=True).to(device)
    
    # Federated training loop
    print("Starting federated training...")
    for round_num in range(1, config['num_rounds'] + 1):
        print(f"\nRound {round_num}/{config['num_rounds']}")
        
        # Randomly select clients for this round (50% participation rate)
        selected_clients = np.random.choice(
            config['num_clients'], 
            size=max(1, int(config['num_clients'] * 0.5)),
            replace=False
        )
        
        # Client update phase
        client_models = []
        client_sizes = []

        for client_idx in selected_clients:
            print(f"\nTraining client {client_idx}...")

            # Create local model copy
            local_model = DistilBARTGen(use_pt_model=True).to(device)
            local_model.load_state_dict(model.state_dict())

            # Get client data with reduced num_workers and pin_memory=False to save memory
            train_loader = DataLoader(
                train_datasets[client_idx],
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config['num_workers'],
                pin_memory=False,
                persistent_workers=False
            )

            # Local training
            optimizer = AdamW(local_model.parameters(), lr=config['learning_rate'])
            local_model.train()

            for epoch in range(config['local_epochs']):
                total_loss = 0
                
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['local_epochs']}"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Forward pass
                    outputs = local_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(local_model.parameters(), config['max_grad_norm'])
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # Calculate and log training metrics
                avg_loss = total_loss / len(train_loader)
                print(f"Client {client_idx} - Epoch {epoch+1}: Loss: {avg_loss:.4f}")
                
                # Log training metrics
                log_metrics(metrics_file, {
                    'round': round_num,
                    'client_id': client_idx,
                    'epoch': epoch + 1,
                    'phase': 'train',
                    'loss': avg_loss,
                    'accuracy': 0.0,  # Not applicable for generation
                    'precision': 0.0,  # Will be updated during evaluation
                    'recall': 0.0,     # Will be updated during evaluation
                    'f1': 0.0          # Will be updated during evaluation
                })
            
            # Store updated model and data size
            client_models.append(local_model.state_dict())
            client_sizes.append(len(train_datasets[client_idx]))
        
        # Aggregate model updates (Federated Averaging)
        print("\nAggregating model updates...")
        global_state = model.state_dict()
        total_size = sum(client_sizes)
        
        # Initialize averaged model parameters
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key])
            
            # Weighted average of client models
            for i, client_state in enumerate(client_models):
                weight = client_sizes[i] / total_size
                global_state[key] += weight * client_state[key]
        
        # Update global model
        model.load_state_dict(global_state)
        
        # Evaluate global model on test set
        print("\nEvaluating global model...")
        model.eval()
        
        # Evaluate on a small subset of the test set for efficiency
        test_loader = DataLoader(
            test_datasets[0],  # Just use first test client for evaluation
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=False,
            persistent_workers=False
        )
        
        # Initialize lists for evaluation
        all_references = []
        all_predictions = []
        total_loss = 0.0
        
        # Evaluate on test set
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass for loss
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()
                
                # Generate summaries
                summary_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
                
                # Decode references and predictions
                preds = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
                refs = [
                    tokenizer.decode([l for l in label if l != -100], skip_special_tokens=True)
                    for label in labels.cpu().numpy()
                ]
                
                all_predictions.extend(preds)
                all_references.extend(refs)
                
                # Print first 2 examples
                if i < 2:
                    article = tokenizer.decode(input_ids[0], skip_special_tokens=True)[:500]
                    print("\nExample", i+1)
                    print("-" * 50)
                    print("ARTICLE:", article + ("..." if len(article) == 500 else ""))
                    print("\nREFERENCE SUMMARY:", refs[0])
                    print("\nGENERATED SUMMARY:", preds[0])
                    print("-" * 50)
        
        # Calculate ROUGE metrics
        avg_loss = total_loss / len(test_loader)
        try:
            if all_predictions and all_references and len(all_predictions) == len(all_references):
                # Ensure we have valid strings for ROUGE calculation
                all_predictions = [str(p) if p is not None else '' for p in all_predictions]
                all_references = [str(r) if r is not None else '' for r in all_references]
                rouge_metrics = calculate_rouge_metrics(all_predictions, all_references)
            else:
                print(f"Warning: Predictions ({len(all_predictions)}) and references ({len(all_references)}) length mismatch or empty")
                rouge_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        except Exception as e:
            print(f"Error calculating ROUGE metrics: {str(e)}")
            print(f"Predictions sample: {all_predictions[:1] if all_predictions else 'None'}")
            print(f"References sample: {all_references[:1] if all_references else 'None'}")
            rouge_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Log evaluation metrics
        log_metrics(metrics_file, {
            'round': round_num,
            'client_id': 'global',
            'epoch': config['local_epochs'],
            'phase': 'eval',
            'loss': avg_loss,
            'accuracy': 0.0,  # Not applicable for generation
            'precision': rouge_metrics['precision'] * 100,  # Convert to percentage
            'recall': rouge_metrics['recall'] * 100,
            'f1': rouge_metrics['f1'] * 100
        })
        
        print(f"\nEvaluation Metrics - Loss: {avg_loss:.4f}, "
              f"ROUGE-L F1: {rouge_metrics['f1']*100:.2f}%")
        
        # Save model checkpoint
        os.makedirs(config['model_save_path'], exist_ok=True)
        checkpoint_path = os.path.join(config['model_save_path'], f'model_round_{round_num}.pt')
        torch.save({
            'round': round_num,
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
            'metrics': {
                'loss': avg_loss,
                **rouge_metrics
            }
        }, checkpoint_path)
        
        print(f"\nModel checkpoint saved to {checkpoint_path}")
        
        # Set back to training mode
        model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DistilBART on CNN/DailyMail with Federated Learning')
    parser.add_argument('--config', type=str, default='configs/distilbart_cnndm.yaml',
                      help='Path to configuration YAML file')
    args = parser.parse_args()
    
    train(config_path=args.config)
