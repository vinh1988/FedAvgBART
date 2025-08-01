import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizerFast

class News20Dataset(Dataset):
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_20newsgroups(data_dir, num_clients=10, test_size=0.2, random_state=42):
    """Load 20 Newsgroups dataset and split it into multiple clients.
    
    Args:
        data_dir (str): Directory to store/load the dataset
        num_clients (int): Number of clients to split the data into
        test_size (float): Fraction of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_datasets, test_datasets, num_classes, tokenizer)
    """
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Load 20 Newsgroups dataset
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    
    # Combine train and test for custom split
    all_texts = np.concatenate([newsgroups_train.data, newsgroups_test.data])
    all_labels = np.concatenate([newsgroups_train.target, newsgroups_test.target])
    
    # Split into train and test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    train_dataset = News20Dataset(train_texts, train_labels, tokenizer)
    test_dataset = News20Dataset(test_texts, test_labels, tokenizer)
    
    # Split into clients using a simple approach (for demonstration)
    # In a real federated learning scenario, you might want to use more sophisticated splitting
    def split_into_clients(dataset, num_clients, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
            
        indices = np.random.permutation(len(dataset))
        client_indices = np.array_split(indices, num_clients)
        
        client_datasets = []
        for client_idx in range(num_clients):
            client_dataset = Subset(dataset, client_indices[client_idx])
            client_datasets.append(client_dataset)
            
        return client_datasets
    
    # Split training and test data among clients
    train_datasets = split_into_clients(train_dataset, num_clients, random_state)
    test_datasets = split_into_clients(test_dataset, num_clients, random_state)
    
    # Get number of classes
    num_classes = len(np.unique(all_labels))
    
    return train_datasets, test_datasets, num_classes, tokenizer

if __name__ == "__main__":
    # Example usage
    data_dir = "./data/20newsgroups"
    train_datasets, test_datasets, num_classes, tokenizer = load_20newsgroups(
        data_dir, num_clients=10, test_size=0.2, random_state=42
    )
    
    print(f"Number of classes: {num_classes}")
    print(f"Number of training clients: {len(train_datasets)}")
    print(f"Number of test clients: {len(test_datasets)}")
    print(f"Sample training data size for client 0: {len(train_datasets[0])}")
    print(f"Sample test data size for client 0: {len(test_datasets[0])}")
