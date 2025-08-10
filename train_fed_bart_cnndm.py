import os
import sys
import torch
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
from rouge_score import rouge_scorer
from datetime import datetime
import yaml
import argparse
import json
from collections import defaultdict
import copy
import torch.nn.functional as F
import gc  # Garbage collection

# Add src to path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.fed_metrics import ClientContributionTracker
from fed_metrics_tracker import MetricsTracker

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bart_gen import BARTGen
from src.datasets.cnndm import load_cnndm

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
        'max_train_samples': 20,  # Set to None to use full dataset
        'max_val_samples': 10,    # Set to None to use full dataset
        'max_test_samples': 10,   # Set to None to use full dataset
        
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
    # Initialize default return values
    default_scores = {
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
    }
    
    # Initialize ROUGE scorer
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Calculate ROUGE scores for each prediction-reference pair
        scores = []
        for pred, ref in zip(predictions, references):
            if pred and ref:  # Only calculate if both prediction and reference are non-empty
                scores.append(scorer.score(ref, pred))
        
        if not scores:
            return default_scores
        
        # Calculate average F1 scores for each ROUGE metric
        avg_rouge1 = sum(s['rouge1'].fmeasure for s in scores) / len(scores)
        avg_rouge2 = sum(s['rouge2'].fmeasure for s in scores) / len(scores)
        avg_rougeL = sum(s['rougeL'].fmeasure for s in scores) / len(scores)
        
        return {
            'rouge1': avg_rouge1,
            'rouge2': avg_rouge2,
            'rougeL': avg_rougeL,
            'precision': avg_rougeL,  # For backward compatibility
            'recall': avg_rougeL,     # For backward compatibility
            'f1': avg_rougeL          # For backward compatibility
        }
        
    except Exception as e:
        print(f"Error calculating ROUGE metrics: {e}")
        return default_scores

def init_metrics_logger(output_dir, experiment_name):
    """
    Initialize metrics logger with consistent CSV format.
    This is kept for backward compatibility but is no longer used.
    The new MetricsTracker class should be used instead.
    
    Args:
        output_dir (str): Directory to save metrics
        experiment_name (str): Name of the experiment
        
    Returns:
        str: Path to the metrics file (deprecated)
    """
    print("Note: init_metrics_logger is deprecated. Using MetricsTracker instead.")
    return os.path.join(output_dir, f"{experiment_name}_metrics_deprecated.csv")

def log_metrics(metrics_file, metrics_dict):
    """
    Log metrics to a CSV file with consistent formatting.
    This is kept for backward compatibility but is no longer used.
    The new MetricsTracker class should be used instead.
    
    Args:
        metrics_file (str): Path to the metrics CSV file
        metrics_dict (dict): Dictionary containing metrics to log
    """
    # Ensure all required fields are present with default values
    default_metrics = {
        'round': 0,
        'phase': '',
        'epoch': 0,
        'client_id': -1,
        'loss': 0.0,
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0,
        'gini_coefficient': 0.0,
        'mean_contribution': 0.0,
        'min_contribution': 0.0,
        'max_contribution': 0.0,
        'cv_contribution': 0.0
    }
    
    # Update with provided metrics
    default_metrics.update(metrics_dict)
    
    # Convert all values to strings and format floats
    formatted_metrics = {}
    for key, value in default_metrics.items():
        if isinstance(value, float):
            formatted_metrics[key] = f"{value:.6f}"
        elif value is None:
            formatted_metrics[key] = ''
        else:
            formatted_metrics[key] = str(value)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(metrics_file)), exist_ok=True)
    
    # Write to CSV
    file_exists = os.path.isfile(metrics_file)
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=formatted_metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(formatted_metrics)

def init_contribution_tracker(output_dir):
    """Initialize client contribution tracker."""
    contribution_file = os.path.join(output_dir, 'client_contributions.csv')
    
    # Create header if file doesn't exist
    if not os.path.exists(contribution_file):
        with open(contribution_file, 'w') as f:
            f.write('round,client_id,contribution,data_size,avg_gradient_norm,update_norm\n')
    
    return contribution_file

def log_client_contribution(contribution_file, round_num, client_id, contribution, data_size, avg_grad_norm, update_norm):
    """Log client contribution metrics to CSV."""
    with open(contribution_file, 'a') as f:
        f.write(f'{round_num},{client_id},{contribution:.6f},{data_size},{avg_grad_norm:.6f},{update_norm:.6f}\n')

def calculate_contribution(initial_weights, updated_weights, global_weights, data_size):
    """
    Calculate client's contribution based on weight updates and data size.
    
    Args:
        initial_weights: Client's model weights before local training
        updated_weights: Client's model weights after local training
        global_weights: Global model weights before aggregation
        data_size: Number of training samples for this client
        
    Returns:
        tuple: (contribution_score, avg_grad_norm, update_norm)
    """
    # Calculate weight update
    update = {}
    for key in initial_weights:
        update[key] = updated_weights[key] - initial_weights[key]
    
    # Calculate L2 norm of the update
    update_norm = 0.0
    for key in update:
        update_norm += torch.norm(update[key].float()).item() ** 2
    update_norm = np.sqrt(update_norm)
    
    # Calculate average gradient norm (approximate as update / learning_rate)
    # This is an approximation since we don't have the actual gradients
    learning_rate = 5e-5  # Should match your learning rate
    avg_grad_norm = update_norm / learning_rate if learning_rate > 0 else 0.0
    
    # Simple contribution score: update norm weighted by data size
    # You can modify this formula based on your specific needs
    contribution_score = update_norm * np.sqrt(data_size)
    
    return contribution_score, avg_grad_norm, update_norm

def save_client_distribution(clients_data, output_dir):
    """Save client data distribution to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    dist_file = os.path.join(output_dir, 'client_data_distribution.csv')
    
    # Convert to DataFrame
    dist_data = []
    
    def get_avg_text_length(dataset, text_key='article'):
        """Helper to get average text length from dataset."""
        if dataset is None or not hasattr(dataset, '__len__') or len(dataset) == 0:
            return 0
            
        try:
            # Handle different dataset formats
            if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
                # Handle Subset objects
                items = []
                for i in dataset.indices:
                    try:
                        item = dataset.dataset[i]
                        if isinstance(item, dict):
                            items.append(item)
                        elif hasattr(item, 'article') and hasattr(item, 'highlights'):
                            items.append({
                                'article': item.article,
                                'highlights': item.highlights
                            })
                    except Exception as e:
                        print(f"Error accessing dataset item {i}: {e}")
                        continue
            elif isinstance(dataset, list):
                items = dataset
            else:
                items = dataset
                
            # Extract text
            texts = []
            for item in items:
                try:
                    if isinstance(item, dict):
                        if text_key in item:
                            texts.append(str(item[text_key]))
                    elif hasattr(item, text_key):
                        texts.append(str(getattr(item, text_key)))
                except (KeyError, AttributeError, IndexError) as e:
                    print(f"Error processing item: {e}")
                    continue
                    
            lengths = [len(t.split()) for t in texts if t]
            return np.mean(lengths) if lengths else 0
            
        except Exception as e:
            print(f"Error calculating text length: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    for client_id, data in clients_data.items():
        try:
            train_samples = len(data['train']) if hasattr(data['train'], '__len__') else 0
            val_samples = len(data['val']) if hasattr(data['val'], '__len__') else 0
            test_samples = len(data['test']) if hasattr(data['test'], '__len__') else 0
            
            dist_data.append({
                'client_id': client_id,
                'train_samples': train_samples,
                'val_samples': val_samples,
                'test_samples': test_samples,
                'avg_train_length': get_avg_text_length(data['train']),
                'avg_val_length': get_avg_text_length(data['val']),
                'avg_test_length': get_avg_text_length(data['test'], 'highlights')
            })
        except Exception as e:
            print(f"Error processing client {client_id}: {e}")
            dist_data.append({
                'client_id': client_id,
                'train_samples': 0,
                'val_samples': 0,
                'test_samples': 0,
                'avg_train_length': 0,
                'avg_val_length': 0,
                'avg_test_length': 0
            })
    
    # Save to CSV
    df = pd.DataFrame(dist_data)
    df.to_csv(dist_file, index=False)
    print(f"Client data distribution saved to {dist_file}")
    
    # Save a summary
    summary_file = os.path.join(output_dir, 'data_distribution_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("=== Data Distribution Summary ===\n\n")
        f.write("Number of clients: {}\n\n".format(len(clients_data)))
        
        f.write("Samples per client:\n")
        f.write(df[['client_id', 'train_samples', 'val_samples', 'test_samples']].to_string(index=False))
        f.write("\n\n")
        
        f.write("Average text length per client:\n")
        f.write(df[['client_id', 'avg_train_length', 'avg_val_length', 'avg_test_length']].to_string(index=False))
        f.write("\n")
    
    print(f"Data distribution summary saved to {summary_file}")
    return dist_file

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
    
    # Initialize metrics tracking
    metrics_tracker = MetricsTracker(
        output_dir=config['output_dir'],
        experiment_name=config['experiment_name'],
        num_clients=config['num_clients']
    )
    
    # Initialize contribution tracker
    contribution_tracker = ClientContributionTracker(
        num_clients=config['num_clients'],
        output_dir=os.path.join(config['output_dir'], 'client_contributions')
    )
    
    # Log system metrics at the start
    metrics_tracker.log_system_metrics()
    print(f"Model checkpoints will be saved to: {config['model_save_path']}")
    print(f"Client contributions will be logged to: {contribution_tracker.output_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset with correct sample sizes
    print("Loading CNN/DailyMail dataset...")
    train_datasets, val_datasets, test_datasets, tokenizer = load_cnndm(
        data_dir=config['data_dir'], 
        num_clients=config['num_clients'], 
        test_size=0.1, 
        random_state=config['seed'],
        max_train_samples=config['max_train_samples'],
        max_val_samples=config.get('max_val_samples', None),
        max_test_samples=config['max_test_samples']
    )
    
    # Save client data distribution
    client_data = {}
    for client_id in range(len(train_datasets)):
        try:
            # Get training data
            train_ds = train_datasets[client_id]
            val_ds = val_datasets[client_id] if val_datasets and client_id < len(val_datasets) else None
            test_ds = test_datasets[client_id] if test_datasets and client_id < len(test_datasets) else None
            
            def get_data(dataset):
                if dataset is None:
                    return []
                if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'data'):
                    return dataset.dataset.data
                elif hasattr(dataset, 'dataset'):
                    return dataset.dataset
                return dataset
            
            client_data[client_id] = {
                'train': get_data(train_ds),
                'val': get_data(val_ds) if val_ds is not None else [],
                'test': get_data(test_ds) if test_ds is not None else []
            }
            
        except Exception as e:
            print(f"Error getting data for client {client_id}: {e}")
            import traceback
            traceback.print_exc()
            client_data[client_id] = {'train': [], 'val': [], 'test': []}
    
    # Save distribution information
    save_client_distribution(client_data, config['output_dir'])
    
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
    model = BARTGen(use_pt_model=True).to(device)
    
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
        client_updates = []
        client_ids = []
        client_data_sizes = []

        for client_idx in selected_clients:
            print(f"\nTraining client {client_idx}...")

            # Create local model copy and save initial weights
            local_model = BARTGen(use_pt_model=True).to(device)
            local_model.load_state_dict(model.state_dict())
            initial_weights = copy.deepcopy(local_model.state_dict())

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
                
                # Log training metrics using the metrics tracker
                metrics_tracker.log_round_metrics(int(round_num), {
                    'client_id': int(client_idx),
                    'epoch': int(epoch + 1),
                    'phase': 'train',
                    'loss': float(avg_loss)
                }, client_id=int(client_idx))
            
            # Calculate client update and store model
            updated_weights = local_model.state_dict()
            client_update = {}
            for key in initial_weights:
                client_update[key] = updated_weights[key] - initial_weights[key]
            
            data_size = len(train_datasets[client_idx])
            
            # Store client update and data size for aggregation
            client_updates.append(client_update)
            client_ids.append(client_idx)
            client_data_sizes.append(data_size)
            
            # Store the full model for aggregation
            client_models.append(updated_weights)
            client_sizes.append(data_size)
            
            print(f"Client {client_idx} - Data Size: {data_size}")
        
        # Log round metrics for all participating clients
        round_metrics = contribution_tracker.log_round(
            round_num=round_num,
            client_updates=client_updates,
            client_sizes=client_data_sizes,
            client_ids=client_ids
        )
        
        if round_metrics:
            print(f"\nRound {round_num} Contribution Metrics:")
            print(f"  Gini Coefficient: {round_metrics['gini']:.4f}")
            print(f"  Mean Contribution: {round_metrics['mean_contribution']:.4f}")
            print(f"  Min/Max Contribution: {round_metrics['min_contribution']:.4f}/{round_metrics['max_contribution']:.4f}")
            print(f"  Coefficient of Variation: {round_metrics['cv_contribution']:.4f}")
            
            # Log contribution metrics using metrics tracker
            metrics_tracker.log_round_metrics(int(round_num), {
                'phase': 'aggregation',
                'gini_coefficient': float(round_metrics['gini']),
                'mean_contribution': float(round_metrics['mean_contribution']),
                'min_contribution': float(round_metrics['min_contribution']),
                'max_contribution': float(round_metrics['max_contribution']),
                'cv_contribution': float(round_metrics['cv_contribution'])
            })
        
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
        
        # Calculate average loss
        avg_loss = total_loss / len(test_loader)
        
        # Ensure we have valid strings for metrics calculation
        all_predictions = [str(p) if p is not None else '' for p in all_predictions]
        all_references = [str(r) if r is not None else '' for r in all_references]
        
        # Calculate all metrics using the metrics tracker
        metrics_dict = metrics_tracker.calculate_all_metrics(all_predictions, all_references)
        
        # Convert all metric values to native Python types
        metrics_dict = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                      for k, v in metrics_dict.items()}
        
        metrics_dict.update({
            'phase': 'evaluation',
            'loss': float(avg_loss)
        })
        
        # Log metrics for this round
        metrics_tracker.log_round_metrics(int(round_num), metrics_dict)
        metrics_tracker.log_system_metrics()
        
        # Print summary
        print(f"\nEvaluation Metrics - Round {round_num}")
        print("-" * 50)
        print(f"Loss: {avg_loss:.4f}")
        
        # Print ROUGE scores if available
        if any(k in metrics_dict for k in ['rouge1_f1', 'rouge1']):
            print("\nROUGE Scores (F1):")
            rouge1 = metrics_dict.get('rouge1_f1', metrics_dict.get('rouge1', 0.0))
            rouge2 = metrics_dict.get('rouge2_f1', metrics_dict.get('rouge2', 0.0))
            rougeL = metrics_dict.get('rougeL_f1', metrics_dict.get('rougeL', 0.0))
            print(f"  ROUGE-1: {rouge1:.4f}")
            print(f"  ROUGE-2: {rouge2:.4f}")
            print(f"  ROUGE-L: {rougeL:.4f}")
        
        # Print BLEU scores
        if any(f'bleu_{n}' in metrics_dict for n in range(1, 5)):
            print("\nBLEU Scores:")
            for n in range(1, 5):
                if f'bleu_{n}' in metrics_dict:
                    print(f"  BLEU-{n}: {metrics_dict[f'bleu_{n}']:.4f}")
        
        # Print other metrics if available
        other_metrics = []
        if 'meteor' in metrics_dict:
            other_metrics.append(f"METEOR: {metrics_dict['meteor']:.4f}")
        if 'bertscore_f1' in metrics_dict:
            other_metrics.append(f"BERTScore F1: {metrics_dict['bertscore_f1']:.4f}")
        
        if other_metrics:
            print("\nOther Metrics:")
            for metric in other_metrics:
                print(f"  {metric}")
        
        print("-" * 50)
        
        # Save model checkpoint
        os.makedirs(config['model_save_path'], exist_ok=True)
        checkpoint_path = os.path.join(config['model_save_path'], f'model_round_{round_num}.pt')
        torch.save({
            'round': round_num,
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
            'metrics': metrics_dict
        }, checkpoint_path)
        
        # Clear GPU cache to free up memory
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"\nModel checkpoint saved to {checkpoint_path}")
        
        # Set back to training mode
        model.train()
    
    # Training complete - save final model and generate contribution summary
    final_model_path = os.path.join(config['model_save_path'], 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'config': config
    }, final_model_path)
    
    # Generate and save final contribution summary
    summary = contribution_tracker.generate_summary()
    
    # Ensure all metrics are saved to disk
    contribution_tracker.save_metrics()
    
    # Generate and save training summary
    training_summary = metrics_tracker.get_summary()
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY".center(60))
    print("="*60)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Total Rounds: {config['num_rounds']}")
    print(f"Number of Clients: {config['num_clients']}")
    print(f"Clients per Round: {config['clients_per_round']}")
    print(f"Total Training Time: {training_summary['elapsed_time_seconds']/60:.2f} minutes")
    
    print("\n" + "-"*60)
    print("CLIENT CONTRIBUTION SUMMARY".center(60))
    print("-"*60)
    print(f"  Overall Gini Coefficient: {summary['inequality']['gini']:.4f}")
    print(f"  Overall Coefficient of Variation: {summary['inequality']['cv']:.4f}")
    
    print("\nParticipation Counts:")
    for client_id, count in sorted(summary['participation'].items(), key=lambda x: x[1], reverse=True):
        print(f"  Client {client_id}: {count} rounds")
    
    print("\nTotal Contributions:")
    for client_id, contrib in sorted(summary['total_contributions'].items(), 
                                   key=lambda x: x[1], reverse=True):
        print(f"  Client {client_id}: {contrib:.4f}")
    
    print("\n" + "="*60)
    print("METRICS SUMMARY".center(60))
    print("="*60)
    
    # Get the best metrics across all rounds
    best_metrics = {}
    for round_num, metrics in metrics_tracker.metrics.items():
        for metric, value in metrics.items():
            if metric not in best_metrics or value > best_metrics[metric][1]:
                best_metrics[metric] = (round_num, value)
    
    # Print best metrics
    print("\nBest Metrics:")
    for metric, (round_num, value) in sorted(best_metrics.items()):
        if metric not in ['round', 'phase']:
            print(f"  {metric.upper()}: {value*100 if metric not in ['loss'] else value:.4f} "
                  f"(Round {round_num})")
    
    print("\n" + "-"*60)
    print(f"\nTraining complete! Final model saved to {final_model_path}")
    print(f"Metrics saved to: {training_summary['metrics_file']}")
    print(f"Client contribution analysis saved to: {contribution_tracker.output_dir}")
    
    # Save final summary to file
    final_summary = {
        'experiment_name': config['experiment_name'],
        'config': config,
        'training_summary': training_summary,
        'client_summary': summary,
        'best_metrics': best_metrics,
        'completion_time': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(config['output_dir'], f"{config['experiment_name']}_final_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"\nFinal summary saved to: {summary_file}")
    print("="*60)
    
    # Generate visualizations
    try:
        from src.visualization.visualize_contributions import plot_contribution_metrics
        plot_contribution_metrics(contribution_tracker.output_dir)
        print(f"Visualizations saved to: {os.path.join(contribution_tracker.output_dir, 'contribution_plots')}")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DistilBART on CNN/DailyMail with Federated Learning')
    parser.add_argument('--config', type=str, default='configs/distilbart_cnndm.yaml',
                      help='Path to configuration YAML file')
    args = parser.parse_args()
    
    train(config_path=args.config)
