import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bart import BART
from src.datasets.news20 import load_20newsgroups, News20Dataset
from transformers import BartTokenizerFast

def parse_args():
    parser = argparse.ArgumentParser(description='Train BART in a federated learning setup on 20 Newsgroups dataset')
    
    # Required parameters
    parser.add_argument('--num_clients', type=int, default=10,
                        help='Number of clients in the federated learning setup')
    parser.add_argument('--num_rounds', type=int, default=22,
                        help='Number of communication rounds')
    parser.add_argument('--epochs_per_client', type=int, default=1,
                        help='Number of local epochs per client')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for the optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for gradient clipping')
    parser.add_argument('--data_dir', type=str, default="./data/20newsgroups",
                        help='Directory containing the 20 Newsgroups dataset')
    parser.add_argument('--model_save_path', type=str, default="./saved_models/bart_20news",
                        help='Path to save the trained model checkpoints')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='Maximum sequence length for tokenization')
    
    return parser.parse_args()

def train():
    # Parse command line arguments
    args = parse_args()
    
    # Configuration
    config = {
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'epochs_per_client': args.epochs_per_client,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_grad_norm': args.max_grad_norm,
        'max_seq_length': args.max_seq_length,
        'model_name': 'bart-20news',
        'project_name': 'federated-bart-20news',
        'data_dir': args.data_dir,
        'model_save_path': args.model_save_path
    }
    
    # Initialize wandb
    wandb.init(
        project=config['project_name'],
        name=f"fed-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=config
    )
    
    # Update config with wandb config (useful for hyperparameter sweeps)
    config = wandb.config
    
    # Unpack config
    num_clients = config['num_clients']
    num_rounds = config['num_rounds']
    epochs_per_client = config['epochs_per_client']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    max_seq_length = config['max_seq_length']
    model_save_path = config['model_save_path']
    
    # Create output directories
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize BART tokenizer
    print("Initializing BART tokenizer...")
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    
    # Load dataset
    print("Loading 20 Newsgroups dataset...")
    train_datasets, test_datasets, num_classes, _ = load_20newsgroups(
        data_dir=config['data_dir'],
        num_clients=num_clients,
        test_size=0.2,
        random_state=42
    )
    
    # Get the original datasets from the first client's subset
    train_dataset = train_datasets[0].dataset
    test_dataset = test_datasets[0].dataset  # Get the first test dataset (all should be the same)
    
    # Create new datasets with BART tokenizer
    new_train_dataset = News20Dataset(
        train_dataset.texts,
        train_dataset.labels,
        tokenizer,
        max_length=max_seq_length
    )
    new_test_dataset = News20Dataset(
        test_dataset.texts,
        test_dataset.labels,
        tokenizer,
        max_length=max_seq_length
    )
    
    # Recreate client datasets with the new tokenizer
    train_datasets = [
        Subset(new_train_dataset, train_datasets[i].indices)
        for i in range(len(train_datasets))
    ]
    test_datasets = [
        Subset(new_test_dataset, test_datasets[i].indices)
        for i in range(len(test_datasets))
    ]
    
    # Create data loaders
    train_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True)
        for ds in train_datasets
    ]
    test_loader = DataLoader(new_test_dataset, batch_size=batch_size, shuffle=False)
    
    # Debug information
    print("\n=== Dataset Information ===")
    print(f"Number of training clients: {len(train_datasets)}")
    print(f"Training samples per client: {[len(ds) for ds in train_datasets]}")
    print(f"Total training samples: {sum(len(ds) for ds in train_datasets)}")
    print(f"Test samples: {len(new_test_dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Check class distribution
    def get_class_distribution(dataset):
        if isinstance(dataset, Subset):
            labels = [dataset.dataset.labels[i] for i in dataset.indices]
        else:
            labels = dataset.labels
        return pd.Series(labels).value_counts().sort_index()
    
    print("\n=== Class Distribution ===")
    print("Training set (first client):")
    print(get_class_distribution(train_datasets[0]))
    print("\nTest set:")
    print(get_class_distribution(new_test_dataset))
    
    # Check for data overlap
    train_texts = set()
    for ds in train_datasets:
        if isinstance(ds, Subset):
            train_texts.update(ds.dataset.texts[i] for i in ds.indices)
        else:
            train_texts.update(ds.texts)
    
    test_texts = set(new_test_dataset.texts)
    overlap = train_texts.intersection(test_texts)
    print(f"\nNumber of overlapping texts between train and test: {len(overlap)}")
    if overlap:
        print("Warning: Found overlapping texts between train and test sets!")
    
    # Print sample data
    print("\n=== Sample Data ===")
    sample = new_train_dataset[0]
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Label: {sample['labels']}")
    print(f"Input text: {tokenizer.decode(sample['input_ids'], skip_special_tokens=True)[:200]}...")
    
    # Initialize global model
    print("Initializing BART model...")
    global_model = BART(num_classes=num_classes, use_pt_model=True).to(device)
    
    # Training loop
    print("Starting federated training...")
    global_round_metrics = []
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n=== Round {round_num}/{num_rounds} ===")
        
        # Client update
        client_models = []
        client_metrics = []
        
        for client_id in range(num_clients):
            print(f"\n--- Client {client_id + 1}/{num_clients} ---")
            
            # Initialize client model with global weights
            client_model = BART(num_classes=num_classes, use_pt_model=True).to(device)
            client_model.load_state_dict(global_model.state_dict())
            client_model.train()
            
            # Optimizer and loss function
            optimizer = AdamW(client_model.parameters(), lr=learning_rate)
            criterion = CrossEntropyLoss()
            
            # Training loop
            for epoch in range(epochs_per_client):
                epoch_loss = 0.0
                all_preds = []
                all_labels = []
                
                for batch in tqdm(train_loaders[client_id], desc=f"Epoch {epoch + 1}/{epochs_per_client}"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Forward pass
                    outputs = client_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                # Calculate metrics
                avg_loss = epoch_loss / len(train_loaders[client_id])
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_labels, all_preds, average='weighted', zero_division=0
                )
                acc = accuracy_score(all_labels, all_preds)
                
                print(f"Client {client_id} - Epoch {epoch + 1}:")
                print(f"  Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                
                # Log metrics for CSV
                client_metrics.append({
                    'round': round_num,
                    'client_id': client_id,
                    'epoch': epoch + 1,
                    'phase': 'train',
                    'loss': avg_loss,
                    'accuracy': acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
                
                # Log to wandb
                wandb.log({
                    'round': round_num,
                    'client': client_id,
                    'epoch': epoch + 1,
                    'train/loss': avg_loss,
                    'train/accuracy': acc,
                    'train/f1': f1,
                    'train/precision': precision,
                    'train/recall': recall
                })
            
            client_models.append(client_model)
        
        # Aggregate client models (simple averaging)
        print("\nAggregating client models...")
        global_weights = {}
        for key in global_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                global_weights[key] = global_model.state_dict()[key].clone()
                continue
                
            # Average weights across clients
            global_weights[key] = torch.stack([
                client_models[i].state_dict()[key].float() for i in range(len(client_models))
            ], 0).mean(0)
        
        # Update global model
        global_model.load_state_dict(global_weights)
        
        # Evaluate global model
        print("\nEvaluating global model...")
        global_model.eval()
        eval_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Debug: Check model mode and parameters
        print(f"Model in training mode: {global_model.training}")
        print(f"Model device: {next(global_model.parameters()).device}")
        print(f"Test loader batches: {len(test_loader)}")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = global_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Handle different output formats
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    eval_loss += outputs.loss.item()
                
                # Get logits from the output
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    # In case the model returns a tuple, the logits are typically the first element
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Get predictions
                preds = torch.argmax(logits, dim=1) if logits.dim() > 1 else logits
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_eval_loss = eval_loss / len(test_loader)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        acc = accuracy_score(all_labels, all_preds)
        
        print(f"\nGlobal Model - Round {round_num}:")
        print(f"  Loss: {avg_eval_loss:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Add validation metrics to client_metrics
        client_metrics.append({
            'round': round_num,
            'client_id': -1,
            'epoch': -1,
            'phase': 'validation',
            'loss': avg_eval_loss,
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
        
        # Log to wandb
        wandb.log({
            'round': round_num,
            'val/loss': avg_eval_loss,
            'val/accuracy': acc,
            'val/f1': f1,
            'val/precision': precision,
            'val/recall': recall
        })
        
        # Save all metrics to CSV after each round
        df_metrics = pd.DataFrame(client_metrics)
        metrics_file = f"results/training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_metrics.to_csv(metrics_file, index=False)
        df_metrics.to_csv("results/training_metrics_latest.csv", index=False)
        print(f"Metrics saved to {metrics_file} and results/training_metrics_latest.csv")
        
        # Save model checkpoint
        if round_num % 5 == 0 or round_num == num_rounds:
            checkpoint_path = os.path.join(model_save_path, f"round_{round_num}")
            global_model.save_pretrained(checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")
            
            # Save model as wandb artifact
            artifact = wandb.Artifact(
                name=f"model-round-{round_num}",
                type="model",
                description=f"BART model after round {round_num}",
                metadata={
                    'round': round_num,
                    'val_loss': avg_eval_loss,
                    'val_accuracy': acc,
                    'val_f1': f1,
                    'val_precision': precision,
                    'val_recall': recall
                }
            )
            artifact.add_dir(checkpoint_path)
            wandb.log_artifact(artifact)
    
    # Save final model
    final_model_path = os.path.join(model_save_path, "final")
    global_model.save_pretrained(final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")
    
    # Save final metrics
    final_metrics_file = os.path.join('results', 'final_metrics.csv')
    df_metrics.to_csv(final_metrics_file, index=False)
    print(f"Final metrics saved to {final_metrics_file}")
    
    # Save metrics as wandb artifact
    metrics_artifact = wandb.Artifact(
        name="training_metrics",
        type="metrics",
        description="Training and validation metrics across all rounds"
    )
    metrics_artifact.add_file(metrics_file)
    metrics_artifact.add_file(final_metrics_file)
    wandb.log_artifact(metrics_artifact)
    
    wandb.finish()

if __name__ == "__main__":
    train()
