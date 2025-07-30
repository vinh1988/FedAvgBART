import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from src.client.bert_client import BERTClient
from src.models.bert_model import BERT
from src.datasets import load_twenty_news_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=3, help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=2, help='Number of federated learning rounds')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of local epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Test batch size')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet distribution parameter')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples to use (None for all)')
    return parser.parse_args()

def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100
    
    # Calculate additional metrics
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    # Per-class metrics
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    
    return avg_loss, accuracy, f1, precision, recall, f1_per_class, precision_per_class, recall_per_class

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("Loading 20 Newsgroups dataset...")
    client_data, test_dataset, num_classes = load_twenty_news_data(
        num_clients=args.num_clients,
        test_size=0.2,
        seed=args.seed,
        alpha=args.alpha,
        max_train_samples=args.max_train_samples
    )
    
    print(f"Initializing global BERT model for {num_classes} classes...")
    global_model = BERT(num_classes=num_classes).to(device)
    
    print(f"Creating {args.num_clients} clients...")
    clients = []
    for i in range(args.num_clients):
        print(f"Initializing client {i} with {len(client_data[i]['texts'])} samples...")
        client = BERTClient(
            client_id=i,
            train_data=client_data[i],
            device=device,
            args=args,
            num_classes=num_classes
        )
        clients.append(client)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False
    )
    
    print("\nStarting federated training...\n")
    
    for round in range(1, args.num_rounds + 1):
        print(f"--- Round {round}/{args.num_rounds} ---\n")
        
        # Train each client
        for client in clients:
            print(f"Training client {client.client_id}...")
            client.train(global_model)
        
        # Federated averaging
        print("\nAveraging client models...")
        with torch.no_grad():
            # Get state dict from the first client's model
            global_state = global_model.state_dict()
            
            # Initialize averaged weights
            avg_weights = {}
            for key in global_state.keys():
                if 'num_batches_tracked' in key:
                    continue
                if global_state[key].dtype in [torch.float32, torch.float16]:
                    avg_weights[key] = torch.zeros_like(global_state[key])
            
            # Sum up weights from all clients
            total_samples = sum(len(client.train_loader.dataset) for client in clients)
            
            for client in clients:
                client_samples = len(client.train_loader.dataset)
                client_weight = client_samples / total_samples
                
                for key in avg_weights.keys():
                    avg_weights[key] += client.model.state_dict()[key] * client_weight
            
            # Update global model
            for key in avg_weights.keys():
                global_state[key] = avg_weights[key]
            
            global_model.load_state_dict(global_state)
            print("Global model updated successfully.\n")
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_loss, accuracy, f1, precision, recall, f1_per_class, precision_per_class, recall_per_class = evaluate_model(
            global_model, test_loader, device
        )
        
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        print("\nPer-class metrics:")
        print(f"Class\tF1\tPrecision\tRecall")
        for i in range(len(f1_per_class)):
            print(f"{i}\t{f1_per_class[i]:.4f}\t{precision_per_class[i]:.4f}\t\t{recall_per_class[i]:.4f}")
        
        print()

if __name__ == "__main__":
    main()
