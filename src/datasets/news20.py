import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

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

def _split_dirichlet_indices(labels, num_clients, alpha, min_size=10, seed=42):
    """Partition indices into num_clients using class-wise Dirichlet with concentration alpha.

    Ensures each client has at least min_size samples (retries with new seeds if needed).
    """
    rs = np.random.RandomState(seed)
    labels = np.array(labels)
    num_classes = int(labels.max()) + 1
    total_indices = np.arange(len(labels))

    def attempt(seed_offset):
        rs_local = np.random.RandomState(seed + seed_offset)
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            c_idx = total_indices[labels == c]
            rs_local.shuffle(c_idx)
            # Dirichlet proportions for this class across clients
            proportions = rs_local.dirichlet([alpha] * num_clients)
            # Convert proportions to split sizes
            sizes = (proportions * len(c_idx)).astype(int)
            # Adjust to match exact count by adding remainder to largest bins
            remainder = len(c_idx) - sizes.sum()
            if remainder > 0:
                # add the remainder to clients with largest fractional parts
                frac = proportions * len(c_idx) - sizes
                for i in np.argsort(-frac)[:remainder]:
                    sizes[i] += 1
            # Now split indices
            start = 0
            for i, sz in enumerate(sizes):
                if sz > 0:
                    client_indices[i].extend(c_idx[start:start+sz])
                    start += sz
        # Shuffle per-client and validate min size
        for i in range(num_clients):
            rs_local.shuffle(client_indices[i])
        if min(len(ci) for ci in client_indices) < min_size:
            return None
        return [np.array(ci) for ci in client_indices]

    for retry in range(50):  # avoid infinite loops
        out = attempt(retry)
        if out is not None:
            return out
    # Fallback: even random split if constraints are too strict
    indices = np.random.RandomState(seed).permutation(len(labels))
    return np.array_split(indices, num_clients)


def load_20newsgroups(
    data_dir,
    num_clients=10,
    test_size=0.2,
    random_state=42,
    dirichlet_alpha=None,
    dirichlet_min_size=10,
    tokenizer_name: str = 'distilbert-base-uncased',
    max_length: int = 128,
):
    """Load 20 Newsgroups dataset and split it into multiple clients.
    
    Args:
        data_dir (str): Directory to store/load the dataset
        num_clients (int): Number of clients to split the data into
        test_size (float): Fraction of data to use for testing
        random_state (int): Random seed for reproducibility
        dirichlet_alpha (float|None): Dirichlet concentration for non-IID split
        dirichlet_min_size (int): Minimum samples per client in Dirichlet split
        tokenizer_name (str): HF tokenizer name (e.g., 'distilbert-base-uncased', 'facebook/bart-large')
        max_length (int): Max sequence length for tokenization
        
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
    
    # Initialize tokenizer (AutoTokenizer for flexibility)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    # Create datasets
    train_dataset = News20Dataset(train_texts, train_labels, tokenizer, max_length=max_length)
    test_dataset = News20Dataset(test_texts, test_labels, tokenizer, max_length=max_length)
    
    # Split into clients: IID random or Dirichlet non-IID
    def to_subsets(dataset, splits):
        return [Subset(dataset, idx) for idx in splits]

    if dirichlet_alpha is None:
        # IID random split
        rs = np.random.RandomState(random_state)
        train_indices = np.array_split(rs.permutation(len(train_dataset)), num_clients)
        test_indices = np.array_split(rs.permutation(len(test_dataset)), num_clients)
    else:
        # Non-IID via Dirichlet on labels
        train_indices = _split_dirichlet_indices(
            labels=train_dataset.labels,
            num_clients=num_clients,
            alpha=float(dirichlet_alpha),
            min_size=int(dirichlet_min_size),
            seed=random_state,
        )
        test_indices = _split_dirichlet_indices(
            labels=test_dataset.labels,
            num_clients=num_clients,
            alpha=float(dirichlet_alpha),
            min_size=max(1, int(dirichlet_min_size/2)),  # allow smaller test shards
            seed=random_state + 1,
        )

    train_datasets = to_subsets(train_dataset, train_indices)
    test_datasets = to_subsets(test_dataset, test_indices)
    
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
