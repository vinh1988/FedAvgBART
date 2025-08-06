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
from transformers import BartForSequenceClassification, BartTokenizerFast
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
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

def train_epoch(model, train_loader, optimizer, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(train_loader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # Get predictions
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    avg_loss = total_loss / len(train_loader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

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
            
            total_loss += outputs.loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    avg_loss = total_loss / len(data_loader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    parser = argparse.ArgumentParser(description='Train BART on 20 Newsgroups (Centralized)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=22, help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--model_save_path', type=str, default='./saved_models/bart_20news_centralized', 
                       help='Path to save the trained model')
    parser.add_argument('--wandb_project', type=str, default='bart-20news-centralized',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_run', type=str, default='run1',
                       help='Weights & Biases run name')
    parser.add_argument('--output_dir', type=str, default='./results_bart_20news_centralized',
                       help='Directory to save metrics and outputs')
    # Always save metrics to CSV
    parser.add_argument('--save_metrics', action='store_true', default=True,
                       help='Save metrics to CSV file (enabled by default)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize metrics storage
    metrics = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_file = os.path.join(args.output_dir, f'training_metrics_centralized_{timestamp}.csv')
    
    # Initialize wandb if enabled
    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_run)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    model_name = 'facebook/bart-base'
    tokenizer = BartTokenizerFast.from_pretrained(model_name)
    model = BartForSequenceClassification.from_pretrained(
        model_name,
        num_labels=20,  # 20 newsgroups
        output_attentions=False,
        output_hidden_states=False,
    ).to(device)
    
    # Load data
    print("Loading data...")
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = load_data()
    
    # Create datasets
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, args.max_length)
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer, args.max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_accuracy = 0.0
    
    print("Starting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Store metrics
        epoch_metrics = {
            'round': epoch + 1,
            'client_id': -1,  # -1 indicates centralized training
            'epoch': epoch + 1,
            'phase': 'train',
            'loss': train_metrics['loss'],
            'accuracy': train_metrics['accuracy'],
            'precision': train_metrics['precision'],
            'recall': train_metrics['recall'],
            'f1': train_metrics['f1'],
            'source_file': f'training_metrics_centralized_{timestamp}.csv'
        }
        metrics.append(epoch_metrics)
        
        val_epoch_metrics = {
            'round': epoch + 1,
            'client_id': -1,
            'epoch': -1,  # -1 for validation
            'phase': 'validation',
            'loss': val_metrics['loss'],
            'accuracy': val_metrics['accuracy'],
            'precision': val_metrics['precision'],
            'recall': val_metrics['recall'],
            'f1': val_metrics['f1'],
            'source_file': f'training_metrics_centralized_{timestamp}.csv'
        }
        metrics.append(val_epoch_metrics)
        
        # Save metrics to CSV after each epoch
        df = pd.DataFrame(metrics)
        df.to_csv(metrics_file, index=False)
        
        # Log to wandb if enabled
        if args.wandb_project:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/precision': train_metrics['precision'],
                'train/recall': train_metrics['recall'],
                'train/f1': train_metrics['f1'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/f1': val_metrics['f1'],
            })
        
        # Save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            model.save_pretrained(args.model_save_path)
            tokenizer.save_pretrained(args.model_save_path)
            print(f"New best model saved with val accuracy: {best_val_accuracy:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    
    # Add test metrics to the metrics list
    test_metrics_entry = {
        'round': args.num_epochs + 1,  # Mark as final round
        'client_id': -1,
        'epoch': -1,
        'phase': 'test',
        'loss': test_metrics['loss'],
        'accuracy': test_metrics['accuracy'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'f1': test_metrics['f1'],
        'source_file': f'training_metrics_centralized_{timestamp}.csv'
    }
    metrics.append(test_metrics_entry)
    
    # Final save of metrics
    df = pd.DataFrame(metrics)
    df.to_csv(metrics_file, index=False)
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Print the path to the saved file
    print(f"CSV file location: {os.path.abspath(metrics_file)}")
    
    # Log test metrics to wandb if enabled
    if args.wandb_project:
        wandb.log({
            'test/loss': test_metrics['loss'],
            'test/accuracy': test_metrics['accuracy'],
            'test/precision': test_metrics['precision'],
            'test/recall': test_metrics['recall'],
            'test/f1': test_metrics['f1'],
        })
        wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    main()
