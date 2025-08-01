import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import random

class CNNDailyMailDataset(Dataset):
    def __init__(self, tokenizer, split='train', max_length=512, max_examples=None):
        """
        Args:
            tokenizer: GPT-2 tokenizer
            split: 'train', 'validation', or 'test'
            max_length: Maximum sequence length
            max_examples: Maximum number of examples to load (None for all)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load CNN/DailyMail dataset
        print(f"Loading CNN/DailyMail {split} dataset...")
        self.dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
        
        if max_examples is not None:
            self.dataset = self.dataset.select(range(min(max_examples, len(self.dataset))))
        
        self.examples = []
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess the dataset and prepare for GPT-2."""
        for example in tqdm(self.dataset, desc="Preprocessing data"):
            article = example['article']
            highlights = example['highlights']
            
            # Format the input as a prompt for summarization
            prompt = f"Summarize the following article:\n\n{article}\n\nSummary:"
            
            # Tokenize the prompt and summary together
            inputs = self.tokenizer(
                prompt,
                highlights,
                max_length=self.max_length,
                truncation='only_first',  # Only truncate the article, not the summary
                padding='max_length',
                return_tensors='pt',
                return_attention_mask=True
            )
            
            # Create labels (shifted right for language modeling)
            labels = inputs['input_ids'].clone()
            # Set labels to -100 for the prompt part so they're ignored in the loss
            prompt_length = len(self.tokenizer(
                prompt, 
                return_tensors='pt',
                padding=False,
                truncation=False
            )['input_ids'][0])
            
            labels[0, :prompt_length] = -100
            
            self.examples.append({
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': labels.squeeze(0)
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def get_cnn_dataloaders(batch_size=4, max_length=512, max_examples=None):
    """Get DataLoaders for CNN/DailyMail dataset with GPT-2 tokenizer."""
    # Initialize GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Add special tokens if they don't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Create datasets
    train_dataset = CNNDailyMailDataset(
        tokenizer, 
        split='train', 
        max_length=max_length,
        max_examples=max_examples
    )
    
    val_dataset = CNNDailyMailDataset(
        tokenizer, 
        split='validation', 
        max_length=max_length,
        max_examples=max_examples
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer
