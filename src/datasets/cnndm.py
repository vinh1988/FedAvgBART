import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from transformers import BartTokenizerFast

class CNNDMDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_length=512, max_target_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # Convert dataset to list to avoid indexing issues
        self.examples = []
        for i in range(len(data)):
            self.examples.append({
                'article': data[i]['article'],
                'highlights': data[i]['highlights']
            })
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Ensure idx is a Python integer
        if isinstance(idx, (np.integer, np.int64)):
            idx = int(idx)
            
        example = self.examples[idx]
        article = example['article']
        highlights = example['highlights']
        
        # Tokenize the inputs and labels
        model_inputs = self.tokenizer(
            article,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                highlights,
                max_length=self.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Replace padding token id with -100 for loss calculation
        labels["input_ids"][labels["input_ids"] == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': model_inputs['input_ids'].squeeze(),
            'attention_mask': model_inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }

def load_cnndm(data_dir, num_clients=10, test_size=0.1, random_state=42, max_train_samples=None, max_test_samples=None):
    """Load CNN/DailyMail dataset and split it into multiple clients.
    
    Args:
        data_dir (str): Directory to store/load the dataset
        num_clients (int): Number of clients to split the data into
        test_size (float): Fraction of data to use for testing
        random_state (int): Random seed for reproducibility
        max_train_samples (int, optional): Maximum number of training samples to use
        max_test_samples (int, optional): Maximum number of test samples to use
        
    Returns:
        tuple: (train_datasets, test_datasets, tokenizer)
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    print("Loading CNN/DailyMail dataset...")
    # Load dataset with streaming to avoid memory issues
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    
    # Process training data
    print("Processing training data...")
    train_data = dataset['train']
    if max_train_samples is not None and len(train_data) > max_train_samples:
        train_indices = np.random.choice(len(train_data), max_train_samples, replace=False)
        train_data = train_data.select(train_indices)
    
    # Process test data
    print("Processing test data...")
    test_data = dataset['test']
    if max_test_samples is not None and len(test_data) > max_test_samples:
        test_indices = np.random.choice(len(test_data), max_test_samples, replace=False)
        test_data = test_data.select(test_indices)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = CNNDMDataset(train_data, tokenizer)
    test_dataset = CNNDMDataset(test_data, tokenizer)
    
    # Split into clients
    def split_into_clients(dataset, num_clients):
        dataset_size = len(dataset)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        
        # Split indices among clients
        client_indices = np.array_split(indices, num_clients)
        
        client_datasets = []
        for client_idx in range(num_clients):
            # Convert to list of Python integers
            client_indices_list = [int(i) for i in client_indices[client_idx]]
            client_dataset = Subset(dataset, client_indices_list)
            client_datasets.append(client_dataset)
            
        return client_datasets
    
    print("Splitting data among clients...")
    train_datasets = split_into_clients(train_dataset, num_clients)
    test_datasets = split_into_clients(test_dataset, num_clients)
    
    print(f"Dataset loaded successfully. {len(train_datasets)} clients with "
          f"{sum(len(ds) for ds in train_datasets)} training examples and "
          f"{sum(len(ds) for ds in test_datasets)} test examples.")
    
    return train_datasets, test_datasets, tokenizer
