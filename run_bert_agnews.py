import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.datasets import load_agnews_data
from src.models import BERT
from src.client import BERTClient
from src.datasets.agnews import AGNewsDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=2, help='Number of federated learning rounds')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of local epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet distribution parameter')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples to use (None for all)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading AG News dataset...")
    client_data, test_data = load_agnews_data(
        num_clients=args.num_clients,
        test_size=args.test_size,
        seed=args.seed,
        alpha=args.alpha,
        max_train_samples=args.max_train_samples
    )
    
    # Create test dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = AGNewsDataset(test_data['texts'], test_data['labels'], tokenizer, max_length=args.max_length)
    
    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize global model
    print("Initializing global BERT model...")
    global_model = BERT(num_classes=4).to(device)
    
    # Create clients
    print(f"Creating {len(client_data)} clients...")
    clients = []
    for client_id, data in client_data.items():
        if len(data['texts']) > 0:  # Only create client if there's data
            print(f"Initializing client {client_id} with {len(data['texts'])} samples...")
            try:
                client = BERTClient(client_id, data, device, args)
                clients.append(client)
            except Exception as e:
                print(f"Error initializing client {client_id}: {str(e)}")
    
    # Federated training
    print("\nStarting federated training...")
    for round_num in range(args.num_rounds):
        print(f"\n--- Round {round_num + 1}/{args.num_rounds} ---")
        
        # Train each client
        client_weights = []
        client_sizes = []
        
        for client in clients:
            print(f"\nTraining client {client.client_id}...")
            try:
                loss, weights = client.train(global_model)
                client_weights.append(weights)
                client_sizes.append(len(client.train_loader.dataset))
                print(f"Client {client.client_id} - Loss: {loss:.4f}, Samples: {client_sizes[-1]}")
            except Exception as e:
                print(f"Error training client {client.client_id}: {str(e)}")
        
        # Federated averaging
        if client_weights and len(client_weights) > 0:
            print("\nAveraging client models...")
            try:
                # Calculate weighted average of client weights
                total_size = sum(client_sizes)
                avg_weights = {}
                
                # Get the state dict from the global model to ensure matching types
                global_state = global_model.state_dict()
                
                # First, identify all unique parameter keys across all clients
                all_keys = set()
                for weights in client_weights:
                    all_keys.update(weights.keys())
                
                for key in all_keys:
                    # Skip non-tensor parameters and buffers that shouldn't be averaged
                    if not any(key in weights for weights in client_weights):
                        continue
                        
                    # Get the first client's tensor to check type and shape
                    first_tensor = next((w[key] for w in client_weights if key in w), None)
                    if not isinstance(first_tensor, torch.Tensor):
                        avg_weights[key] = first_tensor
                        continue
                    
                    # Skip non-parameter tensors that shouldn't be averaged
                    if key.endswith('num_batches_tracked') or not first_tensor.requires_grad:
                        avg_weights[key] = first_tensor.clone()
                        continue
                    
                    # Initialize average tensor with proper type and shape
                    if key in global_state:
                        # Use the global model's tensor as reference for type and device
                        ref_tensor = global_state[key]
                        avg_weights[key] = torch.zeros_like(ref_tensor, device='cpu')
                        
                        # Accumulate weighted sum
                        for i, weights in enumerate(client_weights):
                            if key in weights:
                                weight = float(client_sizes[i] / total_size)
                                client_tensor = weights[key].float().cpu()
                                avg_weights[key].add_(client_tensor * weight)
                        
                        # Ensure correct type and device
                        if ref_tensor.dtype == torch.long:
                            # For long tensors, we need to round to nearest integer
                            avg_weights[key] = avg_weights[key].round().long().to(device=ref_tensor.device)
                        else:
                            # For float tensors, just convert to the correct type
                            avg_weights[key] = avg_weights[key].to(device=ref_tensor.device, 
                                                                dtype=ref_tensor.dtype)
                    else:
                        # If key not in global state, take the first client's weights
                        avg_weights[key] = first_tensor.clone()
                
                # Update global model
                global_model.load_state_dict(avg_weights)
                print("Global model updated successfully.")
            except Exception as e:
                print(f"Error during model averaging: {str(e)}")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        global_model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                try:
                    # Ensure model is on the right device
                    global_model = global_model.to(device)
                    
                    # Forward pass
                    outputs = global_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    # Calculate metrics
                    test_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    
                    # Store predictions and labels for later metrics
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('WARNING: Out of memory during evaluation, skipping batch')
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        # Calculate final metrics
        if len(test_loader) > 0:
            test_loss /= len(test_loader)
            
            # Convert to numpy arrays for metric calculations
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            # Basic metrics
            accuracy = 100 * (all_preds == all_labels).sum() / len(all_labels) if len(all_labels) > 0 else 0.0
            
            # Additional metrics
            from sklearn.metrics import f1_score, precision_score, recall_score
            
            # Calculate metrics for each class and then average
            f1 = f1_score(all_labels, all_preds, average='weighted')
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            # Calculate per-class metrics
            f1_per_class = f1_score(all_labels, all_preds, average=None)
            precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
            recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
            
        else:
            test_loss = float('inf')
            accuracy = 0.0
            f1 = 0.0
            precision = 0.0
            recall = 0.0
        
        # Print metrics
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print("\nPer-class metrics:")
        print(f"Class\tF1\tPrecision\tRecall")
        for i in range(len(f1_per_class)):
            print(f"{i}\t{f1_per_class[i]:.4f}\t{precision_per_class[i]:.4f}\t\t{recall_per_class[i]:.4f}")
        
        # Return metrics for potential logging
        metrics = {
            'test_loss': test_loss,
            'accuracy': accuracy / 100.0,  # Convert back to 0-1 range
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1_per_class.tolist(),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist()
        }

if __name__ == "__main__":
    main()
