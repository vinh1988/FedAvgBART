import os
import torch
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import wandb

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def load_data():
    # Load 20 Newsgroups dataset
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    
    # Split train into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        newsgroups_train.data, 
        newsgroups_train.target,
        test_size=0.2,
        random_state=42,
        stratify=newsgroups_train.target
    )
    
    test_texts = newsgroups_test.data
    test_labels = newsgroups_test.target
    
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

def train_epoch(model, train_loader, optimizer, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(train_loader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss = loss / accumulation_steps  # Normalize loss for accumulation
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps  # Scale back up for logging
        
        # Get predictions
        with torch.no_grad():
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return avg_loss, accuracy, precision, recall, f1

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return avg_loss, accuracy, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description='Train DistilBART on 20 Newsgroups')
    parser.add_argument('--model_name', type=str, default='facebook/bart-base',
                       help='Pretrained model name')
    parser.add_argument('--num_epochs', type=int, default=22,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='./results_distilbart_20news_centralized',
                       help='Directory to save model checkpoints and logs')
    parser.add_argument('--save_metrics', action='store_true',
                       help='Save metrics to CSV file')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for experiment tracking')
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    
    # Initialize Weights & Biases
    if args.use_wandb:
        wandb.init(
            project="distilbart-20news-centralized",
            config={
                "model_name": args.model_name,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "max_seq_length": args.max_seq_length
            }
        )
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=20,  # 20 newsgroups
        output_attentions=False,
        output_hidden_states=False,
    )
    model = model.to(device)
    
    # Load data
    print("Loading data...")
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = load_data()
    
    # Create datasets
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, args.max_seq_length)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, args.max_seq_length)
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer, args.max_seq_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_f1 = 0.0
    metrics_history = []
    
    print("Starting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train for one epoch
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, optimizer, device
        )
        
        # Evaluate on validation set
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, val_loader, device
        )
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "train/precision": train_prec,
                "train/recall": train_rec,
                "train/f1": train_f1,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/precision": val_prec,
                "val/recall": val_rec,
                "val/f1": val_f1,
            })
        
        # Save metrics
        metrics_history.append({
            'epoch': epoch + 1,
            'phase': 'train',
            'loss': train_loss,
            'accuracy': train_acc,
            'precision': train_prec,
            'recall': train_rec,
            'f1': train_f1
        })
        
        metrics_history.append({
            'epoch': epoch + 1,
            'phase': 'validation',
            'loss': val_loss,
            'accuracy': val_acc,
            'precision': val_prec,
            'recall': val_rec,
            'f1': val_f1
        })
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(os.path.join(args.output_dir, 'best_model'))
            tokenizer.save_pretrained(os.path.join(args.output_dir, 'best_model'))
            print(f"New best model saved with Val F1: {best_val_f1:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
        model, test_loader, device
    )
    print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    # Save final model
    model.save_pretrained(os.path.join(args.output_dir, 'final_model'))
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'final_model'))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for the metrics file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(args.output_dir, f'training_metrics_centralized_{timestamp}.csv')
    
    # Prepare metrics data in the required format
    metrics_data = []
    
    # Process training and validation metrics
    for i in range(len(metrics_history)):
        if i % 2 == 0:  # Training metrics
            epoch = (i // 2) + 1
            phase = 'train'
        else:  # Validation metrics
            epoch = -1  # -1 for validation as per the format
            phase = 'validation'
            
        metrics_data.append({
            'round': (i // 2) + 1,
            'client_id': -1,  # -1 for centralized training
            'epoch': epoch,
            'phase': phase,
            'loss': metrics_history[i]['loss'],
            'accuracy': metrics_history[i]['accuracy'],
            'precision': metrics_history[i]['precision'],
            'recall': metrics_history[i]['recall'],
            'f1': metrics_history[i]['f1'],
            'source_file': os.path.basename(metrics_file)
        })
    
    # Add test metrics
    metrics_data.append({
        'round': args.num_epochs,
        'client_id': -1,
        'epoch': args.num_epochs,
        'phase': 'test',
        'loss': test_loss,
        'accuracy': test_acc,
        'precision': test_prec,
        'recall': test_rec,
        'f1': test_f1,
        'source_file': os.path.basename(metrics_file)
    })
    
    # Convert to DataFrame and save
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")
    
    # Also save a copy of metrics in the format expected by visualize_centralized.py
    metrics_df.to_csv(os.path.join(args.output_dir, 'latest_metrics.csv'), index=False)
    
    # Log test metrics to wandb
    if args.use_wandb:
        wandb.log({
            "test/loss": test_loss,
            "test/accuracy": test_acc,
            "test/precision": test_prec,
            "test/recall": test_rec,
            "test/f1": test_f1,
        })
        wandb.finish()
    
    print("Training completed!")

if __name__ == "__main__":
    main()
