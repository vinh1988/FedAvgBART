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

def load_cnndm(data_dir, num_clients=10, test_size=0.1, random_state=42,
               max_train_samples=None, max_val_samples=None, max_test_samples=None,
               dirichlet_alpha=None):
    """Load CNN/DailyMail dataset and split it into multiple clients.
    
    Args:
        data_dir (str): Directory to store/load the dataset
        num_clients (int): Number of clients to split the data into
        test_size (float): Fraction of data to use for testing
        random_state (int): Random seed for reproducibility
        max_train_samples (int, optional): Maximum number of training samples per client
        max_val_samples (int, optional): Maximum number of validation samples per client
        max_test_samples (int, optional): Maximum number of test samples per client
        
    Returns:
        tuple: (train_datasets, val_datasets, test_datasets, tokenizer)
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    print("Loading CNN/DailyMail dataset...")
    # Load dataset with streaming to avoid memory issues
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
    
    def process_split(split_name, max_samples=None):
        print(f"Processing {split_name} data...")
        data = dataset[split_name]
        if max_samples is not None and len(data) > max_samples:
            # If we have more samples than the max, randomly select max_samples
            indices = np.random.choice(len(data), max_samples, replace=False)
            data = data.select(indices)
        return data
    
    # Process data splits
    train_data = process_split('train', max_train_samples)
    val_data = process_split('validation', max_val_samples)
    test_data = process_split('test', max_test_samples)
    
    def create_dataset(data):
        return CNNDMDataset(data, tokenizer) if len(data) > 0 else None
    
    # Create dataset objects
    print("Creating dataset objects...")
    train_dataset = create_dataset(train_data)
    val_dataset = create_dataset(val_data)
    test_dataset = create_dataset(test_data)
    
    def split_into_clients(dataset, num_clients, max_samples_per_client=None):
        if dataset is None:
            return [None] * num_clients
            
        dataset_size = len(dataset)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        
        # Calculate how many samples to use in total
        if max_samples_per_client is not None:
            max_total = min(max_samples_per_client * num_clients, dataset_size)
        else:
            max_total = dataset_size
        
        # Limit the indices to max_total
        if len(indices) > max_total:
            indices = indices[:max_total]

        # Determine client slice sizes
        if dirichlet_alpha is not None:
            # Sample proportions and convert to integer counts summing to max_total
            props = np.random.dirichlet(alpha=[float(dirichlet_alpha)] * num_clients)
            counts = np.maximum(1, np.round(props * len(indices)).astype(int))
            # Adjust to match exactly
            diff = int(np.sum(counts) - len(indices))
            if diff != 0:
                # Trim or pad the largest counts to fix diff
                order = np.argsort(-counts)
                i = 0
                while diff > 0 and i < len(order):
                    if counts[order[i]] > 1:
                        counts[order[i]] -= 1
                        diff -= 1
                    i = (i + 1) % len(order)
                while diff < 0:
                    counts[order[0]] += 1
                    diff += 1
            # Create client index slices
            client_indices = []
            start = 0
            for c in counts:
                end = start + int(c)
                client_indices.append(indices[start:end])
                start = end
        else:
            # IID-ish equal split
            client_indices = np.array_split(indices, num_clients)
        
        client_datasets = []
        for client_idx in range(num_clients):
            if client_idx < len(client_indices):
                # Convert to list of Python integers and apply per-client limit
                client_indices_list = [int(i) for i in client_indices[client_idx] if i < len(dataset)]
                if max_samples_per_client is not None and len(client_indices_list) > max_samples_per_client:
                    client_indices_list = client_indices_list[:max_samples_per_client]
                client_dataset = Subset(dataset, client_indices_list) if client_indices_list else None
            else:
                client_dataset = None
            client_datasets.append(client_dataset)
            
        return client_datasets
    
    print("Splitting data among clients...")
    train_datasets = split_into_clients(train_dataset, num_clients, max_samples_per_client=max_train_samples)
    val_datasets = split_into_clients(val_dataset, num_clients, max_samples_per_client=max_val_samples)
    test_datasets = split_into_clients(test_dataset, num_clients, max_samples_per_client=max_test_samples)
    
    # Debug logging for dataset sizes
    print("\nDebug - Dataset sizes after splitting:")
    print(f"- Training datasets: {[len(ds) if ds is not None else 0 for ds in train_datasets]}")
    print(f"- Validation datasets: {[len(ds) if ds is not None else 0 for ds in val_datasets]}")
    print(f"- Test datasets: {[len(ds) if ds is not None else 0 for ds in test_datasets]}")
    
    # Check for None datasets
    print("\nDebug - Number of None datasets:")
    print(f"- Training: {sum(1 for ds in train_datasets if ds is None)}/{len(train_datasets)}")
    print(f"- Validation: {sum(1 for ds in val_datasets if ds is None)}/{len(val_datasets)}")
    print(f"- Test: {sum(1 for ds in test_datasets if ds is None)}/{len(test_datasets)}")
    
    print(f"Dataset loaded successfully. {num_clients} clients with:"
          f"\n- Training: {sum(len(ds) if ds else 0 for ds in train_datasets)} total samples"
          f"\n- Validation: {sum(len(ds) if ds else 0 for ds in val_datasets)} total samples"
          f"\n- Test: {sum(len(ds) if ds else 0 for ds in test_datasets)} total samples")
    
    return train_datasets, val_datasets, test_datasets, tokenizer
