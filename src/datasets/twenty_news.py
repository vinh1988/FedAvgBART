import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

class TwentyNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_twenty_news_data(num_clients=10, test_size=0.2, seed=42, alpha=0.5, max_train_samples=None):
    """Load 20 Newsgroups dataset and split into clients using Dirichlet distribution."""
    # Load 20 Newsgroups dataset
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    
    # Convert to DataFrame for easier manipulation
    train_texts = newsgroups_train.data
    train_labels = newsgroups_train.target
    
    test_texts = newsgroups_test.data
    test_labels = newsgroups_test.target
    
    print(f"Using {len(train_texts):,} training samples and {len(test_texts):,} test samples")
    
    # Limit samples if specified
    if max_train_samples is not None and max_train_samples < len(train_texts):
        train_texts = train_texts[:max_train_samples]
        train_labels = train_labels[:max_train_samples]
    
    # Split train data among clients using Dirichlet distribution
    n_classes = len(set(train_labels))
    client_data = {i: {'texts': [], 'labels': []} for i in range(num_clients)}
    
    for c in range(n_classes):
        # Get indices of samples with class c
        idx = [i for i, l in enumerate(train_labels) if l == c]
        if not idx:
            continue
            
        # Split class samples among clients using Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Ensure we don't split into more parts than we have samples
        num_samples = len(idx)
        if num_samples < num_clients:
            # If fewer samples than clients, distribute samples to first n clients
            for i in range(num_samples):
                client_data[i % num_clients]['texts'].append(train_texts[idx[i]])
                client_data[i % num_clients]['labels'].append(train_labels[idx[i]])
            continue
            
        # Calculate split points
        split_points = (np.cumsum(proportions) * num_samples).astype(int)[:-1]
        idx_splits = np.split(np.random.permutation(idx), split_points)
        for i in range(num_clients):
            if i < len(idx_splits):
                client_data[i]['texts'].extend([train_texts[j] for j in idx_splits[i]])
                client_data[i]['labels'].extend([train_labels[j] for j in idx_splits[i]])
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create test dataset
    test_dataset = TwentyNewsDataset(
        test_texts,
        test_labels,
        tokenizer=tokenizer,
        max_length=128
    )
    
    return client_data, test_dataset, n_classes
