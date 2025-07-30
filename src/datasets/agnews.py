import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

class AGNewsDataset(Dataset):
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
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_agnews_data(num_clients=10, test_size=0.2, seed=42, alpha=0.5, max_train_samples=5000):
    """Load AG News dataset and split into clients using Dirichlet distribution."""
    # Load AG News dataset
    train_df = pd.read_csv(
        'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv',
        header=None, 
        names=['class', 'title', 'description']
    )
    
    # Limit the number of samples if specified
    if max_train_samples is not None and max_train_samples < len(train_df):
        train_df = train_df.sample(n=max_train_samples, random_state=seed)
    
    # Combine title and description
    train_df['text'] = train_df['title'] + ' ' + train_df['description']
    train_df['label'] = train_df['class'] - 1
    
    # Use the full test set
    test_df = pd.read_csv(
        'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv',
        header=None, 
        names=['class', 'title', 'description']
    )
    test_df['text'] = test_df['title'] + ' ' + test_df['description']
    test_df['label'] = test_df['class'] - 1
    
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    print(f"Using {len(train_texts):,} training samples and {len(test_texts):,} test samples")
    
    # Split train data among clients using Dirichlet distribution
    n_classes = len(set(train_labels))
    client_data = {i: {'texts': [], 'labels': []} for i in range(num_clients)}
    
    for c in range(n_classes):
        # Get indices of samples with this class
        idx = np.where(np.array(train_labels) == c)[0]
        np.random.shuffle(idx)
        
        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        split_idx = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        
        # Split the indices according to the proportions
        idx_splits = np.split(idx, split_idx)
        
        # Assign to clients
        for client_idx, idx_split in enumerate(idx_splits):
            client_data[client_idx]['texts'].extend([train_texts[i] for i in idx_split])
            client_data[client_idx]['labels'].extend([train_labels[i] for i in idx_split])
    
    # Remove clients with no data
    client_data = {k: v for k, v in client_data.items() if len(v['texts']) > 0}
    
    # Create test data dictionary
    test_data = {
        'texts': test_texts,
        'labels': test_labels
    }
    
    return client_data, test_data
