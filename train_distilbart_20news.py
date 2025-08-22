import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.distilbart import DistilBART
from src.datasets.news20 import load_20newsgroups

def parse_args():
    parser = argparse.ArgumentParser(description="Federated DistilBART on 20 Newsgroups")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=22, help="Federated rounds")
    parser.add_argument("--participation_rate", type=float, default=0.5,
                        help="Fraction of clients selected per round (0 < r <= 1)")
    parser.add_argument("--min_clients_per_round", type=int, default=1,
                        help="Minimum clients selected per round (use 2 for small-N)")
    parser.add_argument("--dirichlet_alpha", type=float, default=None,
                        help="Dirichlet concentration for non-IID split; None means IID random split")
    parser.add_argument("--dirichlet_min_size", type=int, default=10,
                        help="Minimum samples per client in Dirichlet partition")
    parser.add_argument("--output_dir", type=str, default="results_distilbart_fed_runs_20news",
                        help="Base directory to store run artifacts (metrics, checkpoints, logs)")
    return parser.parse_args()

def train(args):
    # Configuration
    config = {
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'participation_rate': args.participation_rate,
        'min_clients_per_round': args.min_clients_per_round,
        'dirichlet_alpha': args.dirichlet_alpha,
        'dirichlet_min_size': args.dirichlet_min_size,
        'epochs_per_client': 1,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'max_seq_length': 128,
        'model_name': 'distilbart-20news',
        'project_name': 'federated-distilbart-20news',
        'data_dir': "./data/20newsgroups",
        'model_save_path': "./saved_models/distilbart_20news",
        'output_dir': args.output_dir,
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
    participation_rate = float(config.get('participation_rate', 0.5))
    min_clients_per_round = int(config.get('min_clients_per_round', 1))
    dirichlet_alpha = config.get('dirichlet_alpha', None)
    dirichlet_min_size = int(config.get('dirichlet_min_size', 10))
    epochs_per_client = config['epochs_per_client']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    max_seq_length = config['max_seq_length']
    data_dir = config['data_dir']
    model_save_path = config['model_save_path']
    
    # Create run-specific results directory
    output_dir = config.get('output_dir', 'fed_runs')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        output_dir,
        f"clients_{num_clients}_rounds_{num_rounds}",
        timestamp,
    )
    os.makedirs(run_dir, exist_ok=True)
    results_file = os.path.join(run_dir, f"training_metrics_{timestamp}.csv")
    
    # Initialize metrics storage
    metrics_columns = [
        'round', 'client_id', 'epoch', 'phase',
        'loss', 'accuracy', 'precision', 'recall', 'f1'
    ]
    metrics_data = []
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading 20 Newsgroups dataset...")
    train_datasets, test_datasets, num_classes, tokenizer = load_20newsgroups(
        data_dir,
        num_clients=num_clients,
        test_size=0.2,
        random_state=42,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_min_size=dirichlet_min_size,
    )
    # Compute and save per-client data contributions and label distributions
    total_train = sum(len(ds) for ds in train_datasets)
    total_test = sum(len(ds) for ds in test_datasets)
    # Access base label arrays from underlying datasets
    base_train_labels = train_datasets[0].dataset.labels
    base_test_labels = test_datasets[0].dataset.labels

    contrib_rows = []
    label_rows_train = []
    label_rows_test = []
    for cid in range(num_clients):
        tr_ds = train_datasets[cid]
        te_ds = test_datasets[cid]
        tr_idx = tr_ds.indices
        te_idx = te_ds.indices
        tr_size = len(tr_ds)
        te_size = len(te_ds)
        contrib_rows.append({
            'client_id': cid,
            'train_size': tr_size,
            'train_prop': tr_size / total_train if total_train > 0 else 0.0,
            'test_size': te_size,
            'test_prop': te_size / total_test if total_test > 0 else 0.0,
        })
        # Label distributions
        tr_labels = base_train_labels[tr_idx]
        te_labels = base_test_labels[te_idx]
        tr_counts = np.bincount(tr_labels, minlength=num_classes)
        te_counts = np.bincount(te_labels, minlength=num_classes)
        for cls in range(num_classes):
            label_rows_train.append({'client_id': cid, 'class_id': cls, 'count': int(tr_counts[cls])})
            label_rows_test.append({'client_id': cid, 'class_id': cls, 'count': int(te_counts[cls])})

    pd.DataFrame(contrib_rows).to_csv(os.path.join(run_dir, 'data_contribution.csv'), index=False)
    pd.DataFrame(label_rows_train).to_csv(os.path.join(run_dir, 'label_distribution_train.csv'), index=False)
    pd.DataFrame(label_rows_test).to_csv(os.path.join(run_dir, 'label_distribution_test.csv'), index=False)
    
    # Initialize global model
    print("Initializing DistilBART model...")
    model = DistilBART(
        num_classes=num_classes,
        num_embeddings=tokenizer.vocab_size,
        embedding_size=768,  # DistilBERT base dimension
        hidden_size=768,
        dropout=0.1,
        use_pt_model=True,  # Use pre-trained weights
        is_seq2seq=False
    ).to(device)
    
    # Define loss function and optimizer
    criterion = CrossEntropyLoss()
    
    # Federated training loop
    print("Starting federated training...")
    participation_rows = []
    for round_num in range(1, num_rounds + 1):
        print(f"\nRound {round_num}/{num_rounds}")
        
        # Randomly select clients for this round based on participation settings
        # Compute number of participants using ceil, enforce bounds
        k = int(np.ceil(num_clients * participation_rate))
        k = max(min_clients_per_round, k)
        k = min(k, num_clients)
        selected_clients = np.random.choice(
            num_clients,
            size=k,
            replace=False
        )
        
        # Record participation for this round
        for cid in range(num_clients):
            participation_rows.append({'round': round_num, 'client_id': cid, 'selected': int(cid in selected_clients)})

        # Client update phase
        client_models = []
        client_sizes = []
        
        for client_idx in selected_clients:
            print(f"\nTraining client {client_idx}...")
            
            # Create local model copy
            local_model = DistilBART(
                num_classes=num_classes,
                num_embeddings=tokenizer.vocab_size,
                embedding_size=768,
                hidden_size=768,
                dropout=0.1,
                use_pt_model=True,
                is_seq2seq=False
            ).to(device)
            local_model.load_state_dict(model.state_dict())
            
            # Get client data
            train_dataset = train_datasets[client_idx]
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            # Local training
            optimizer = AdamW(local_model.parameters(), lr=learning_rate)
            local_model.train()
            
            for epoch in range(epochs_per_client):
                total_loss = 0
                correct = 0
                total = 0
                
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_per_client}"):
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
                    logits = outputs.logits
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Calculate metrics
                    total_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                # Calculate metrics
                avg_loss = total_loss / len(train_loader)
                accuracy = 100 * correct / total
                
                # Get predictions and labels for this epoch
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for batch in train_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        _, predicted = torch.max(outputs.logits, 1)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                # Calculate precision, recall, f1 for training
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_labels, all_preds, average='weighted', zero_division=0
                )
                
                # Log metrics to wandb
                wandb.log({
                    'round': round_num,
                    'client': client_idx,
                    'epoch': epoch + 1,
                    'train/loss': avg_loss,
                    'train/accuracy': accuracy / 100,
                    'train/precision': precision,
                    'train/recall': recall,
                    'train/f1': f1
                }, step=round_num)
                
                # Store training metrics
                metrics_data.append({
                    'round': round_num,
                    'client_id': client_idx,
                    'epoch': epoch + 1,
                    'phase': 'train',
                    'loss': avg_loss,
                    'accuracy': accuracy / 100,  # Convert to 0-1 range
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
                
                print(f"Client {client_idx} - Epoch {epoch+1}: "
                      f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
            
            # Store updated model and data size
            client_models.append(local_model.state_dict())
            client_sizes.append(len(train_dataset))
        
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
        
        # Initialize metrics for validation
        all_preds = []
        all_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for client_idx in range(num_clients):  # Evaluate on all clients
                test_loader = DataLoader(
                    test_datasets[client_idx],
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
                
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
                    _, predicted = torch.max(outputs.logits, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss /= len(all_preds)  # Average loss per sample
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Log validation metrics to wandb
        wandb.log({
            'round': round_num,
            'val/loss': val_loss,
            'val/accuracy': accuracy,
            'val/precision': precision,
            'val/recall': recall,
            'val/f1': f1
        }, step=round_num)
        
        # Store validation metrics
        metrics_data.append({
            'round': round_num,
            'client_id': -1,  # -1 indicates global model evaluation
            'epoch': epochs_per_client,  # Last epoch of the round
            'phase': 'validation',
            'loss': val_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        print(f"Round {round_num} - Validation: "
              f"Loss: {val_loss:.4f}, Acc: {accuracy*100:.2f}%, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Save metrics to CSV after each round
        metrics_df = pd.DataFrame(metrics_data, columns=metrics_columns)
        metrics_df.to_csv(results_file, index=False)
        print(f"Metrics saved to {results_file}")
        
        # Save model checkpoint into run-specific folder
        model_checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(model_checkpoint_dir, exist_ok=True)
        torch.save({
            'round': round_num,
            'model_state_dict': model.state_dict(),
            'accuracy': accuracy,
            'loss': val_loss,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }, os.path.join(model_checkpoint_dir, f'model_round_{round_num}.pt'))
        
        print(f"Model checkpoint saved to {model_checkpoint_dir}")
        
        # Save model checkpoints to wandb
        checkpoint_path = os.path.join(model_checkpoint_dir, f'model_round_{round_num}.pt')
        wandb.save(checkpoint_path)
    
    # Save final metrics to a CSV file
    metrics_df = pd.DataFrame(metrics_data)
    metrics_file = os.path.join(run_dir, f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    metrics_df.to_csv(metrics_file, index=False)
    wandb.save(metrics_file)
    # Save participation log
    if participation_rows:
        pd.DataFrame(participation_rows).to_csv(os.path.join(run_dir, 'participation.csv'), index=False)
    
    # Log the final metrics as a table
    metrics_table = wandb.Table(dataframe=metrics_df)
    wandb.log({"metrics_table": metrics_table})
    
    # Finish the wandb run
    wandb.finish()

    # Final metrics summary
    print("\nTraining completed. Final metrics summary:")
    metrics_df = pd.read_csv(results_file)
    print(metrics_df.groupby(['phase']).mean(numeric_only=True)[['loss', 'accuracy', 'precision', 'recall', 'f1']])

if __name__ == "__main__":
    args = parse_args()
    train(args)
