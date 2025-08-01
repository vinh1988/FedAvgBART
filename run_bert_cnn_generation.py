import os
import os
import random
import time
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    get_linear_schedule_with_warmup,
    AdamW,
    set_seed
)

from src.datasets.cnn_dailymail import get_cnn_dataloaders

def save_training_stats(stats, filename):
    """Save training statistics to a JSON file."""
    import json
    with open(filename, 'w') as f:
        json.dump(stats, f, indent=2)

def train_epoch(model, dataloader, optimizer, scheduler, device, config, scaler=None):
    """Train the model for one epoch with gradient accumulation and mixed precision."""
    model.train()
    total_loss = 0
    total_steps = len(dataloader)
    
    # Initialize gradient accumulation
    optimizer.zero_grad()
    
    # Initialize progress bar
    progress_bar = tqdm(
        enumerate(dataloader), 
        total=total_steps,
        desc=f"Training"
    )
    
    for step, batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass with mixed precision
        with autocast(enabled=config['fp16']):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Scale loss for gradient accumulation
            loss = outputs.loss / config['gradient_accumulation_steps']
        
        # Backward pass with gradient scaling if using mixed precision
        if config['fp16']:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += outputs.loss.item()
        
        # Perform optimization step after accumulating enough gradients
        if (step + 1) % config['gradient_accumulation_steps'] == 0 or step == total_steps - 1:
            # Gradient clipping
            if config['fp16']:
                scaler.unscale_(optimizer)
                
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['max_grad_norm']
            )
            
            # Optimizer step with gradient scaling if using mixed precision
            if config['fp16']:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            # Update learning rate
            scheduler.step()
            
            # Clear gradients
            optimizer.zero_grad()
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = total_loss / (step + 1)
        progress_bar.set_postfix({
            'loss': f"{loss.item() * config['gradient_accumulation_steps']:.4f}",
            'avg_loss': f"{avg_loss:.4f}",
            'lr': f"{current_lr:.2e}"
        })
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, tokenizer, max_examples=3):
    """
    Evaluate the model on the validation set and generate some example summaries.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the validation set
        device: Device to run evaluation on
        tokenizer: Tokenizer for decoding
        max_examples: Maximum number of examples to generate and show
        
    Returns:
        tuple: (average_loss, rouge_scores)
    """
    model.eval()
    total_loss = 0
    
    # Initialize ROUGE scorer
    try:
        from rouge_score import rouge_scorer
        rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        calculate_rouge = True
    except ImportError:
        print("ROUGE not available. Install with: pip install rouge-score")
        calculate_rouge = False
    
    # For storing ROUGE scores
    rouge_scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
    }
    
    # For storing example generations
    examples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Calculate loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()
            
            # Generate summaries for ROUGE calculation
            if calculate_rouge and len(examples) < max_examples:
                # Get the first example in the batch
                example_input = input_ids[0].unsqueeze(0)
                
                # Get reference summary (from the input prompt)
                input_text = tokenizer.decode(example_input[0], skip_special_tokens=True)
                
                # Extract the actual reference summary (everything after "Summary:")
                if "Summary:" in input_text:
                    # Split on "Summary:" and take everything after it
                    parts = input_text.split("Summary:")
                    article = parts[0].strip()
                    reference = parts[1].strip() if len(parts) > 1 else ""
                else:
                    article = input_text
                    reference = ""
                
                # Generate summary
                summary_ids = model.generate(
                    example_input,
                    max_length=150,
                    min_length=30,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Decode the generated summary
                generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                # Calculate ROUGE scores if we have a reference
                if reference:
                    scores = rouge_scorer.score(reference, generated_summary)
                    
                    # Store the scores
                    for metric in ['rouge1', 'rouge2', 'rougeL']:
                        rouge_scores[metric]['precision'].append(scores[metric].precision)
                        rouge_scores[metric]['recall'].append(scores[metric].recall)
                        rouge_scores[metric]['fmeasure'].append(scores[metric].fmeasure)
                
                # Store the example
                examples.append({
                    'article': article,
                    'reference': reference,
                    'generated': generated_summary
                })
    
    # Calculate average ROUGE scores
    avg_rouge = {}
    if calculate_rouge and any(len(scores['fmeasure']) > 0 for scores in rouge_scores.values()):
        for metric in rouge_scores:
            if rouge_scores[metric]['fmeasure']:  # Only if we have scores
                avg_rouge[f"{metric}_precision"] = np.mean(rouge_scores[metric]['precision'])
                avg_rouge[f"{metric}_recall"] = np.mean(rouge_scores[metric]['recall'])
                avg_rouge[f"{metric}_f1"] = np.mean(rouge_scores[metric]['fmeasure'])
    
    # Print examples
    if examples:
        print("\n" + "="*80)
        print("EXAMPLE GENERATIONS")
        print("="*80)
        
        for i, example in enumerate(examples):
            print(f"\nEXAMPLE {i+1}:")
            print("-" * 80)
            print("ARTICLE:")
            print(example['article'][:500] + ("..." if len(example['article']) > 500 else ""))
            
            if example['reference']:
                print("\nREFERENCE SUMMARY:")
                print(example['reference'])
            
            print("\nGENERATED SUMMARY:")
            print(example['generated'])
            
            # Print ROUGE scores if available
            if calculate_rouge and i < len(rouge_scores['rouge1']['fmeasure']):
                print("\nROUGE SCORES:")
                for metric in ['rouge1', 'rouge2', 'rougeL']:
                    if rouge_scores[metric]['fmeasure']:  # Only if we have scores
                        print(f"  {metric.upper()}: P={rouge_scores[metric]['precision'][i]:.4f} "
                              f"R={rouge_scores[metric]['recall'][i]:.4f} "
                              f"F1={rouge_scores[metric]['fmeasure'][i]:.4f}")
            
            print("-" * 80)
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, avg_rouge

def generate_summary(model, tokenizer, article, device, max_length=150, num_beams=4):
    """Generate a summary for the given article using GPT-2."""
    model.eval()
    
    # Truncate article if it's too long
    max_article_length = 400  # Reduced to leave room for the summary
    article = ' '.join(article.split()[:max_article_length])
    
    # Create prompt for summarization
    prompt = f"Summarize the following article:\n\n{article}\n\nSummary:"
    
    # Tokenize the input
    inputs = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        return_tensors='pt',
        padding='max_length'
    ).to(device)
    
    # Generate summary
    with torch.no_grad():
        try:
            summary_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length + len(inputs['input_ids'][0]),
                min_length=30 + len(inputs['input_ids'][0]),
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode the generated text and remove the prompt
            full_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summary = full_text[len(prompt):].strip()
            
            # Clean up any incomplete sentences at the end
            last_period = summary.rfind('.')
            if last_period > 0:
                summary = summary[:last_period + 1]
                
            return summary if summary else "No summary generated"
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return "Error generating summary"

def main():
    # Configuration
    config = {
        'batch_size': 2,  # Reduced batch size for GPT-2
        'max_length': 512,
        'learning_rate': 5e-5,  # Adjusted learning rate for GPT-2
        'weight_decay': 0.01,   # Weight decay for regularization
        'num_epochs': 5,        # Increased number of epochs
        'warmup_ratio': 0.1,    # Ratio of warmup steps
        'max_grad_norm': 1.0,   # Gradient clipping
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'saved_models',
        'max_examples': 1000,   # Set to None to use full dataset
        'log_interval': 10,     # Log every N batches
        'save_interval': 1,     # Save model every N epochs
        'gradient_accumulation_steps': 4,  # Accumulate gradients
        'fp16': True,           # Enable mixed precision training
        'seed': 42,             # Random seed
    }
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Create output directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, tokenizer = get_cnn_dataloaders(
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        max_examples=config['max_examples']
    )
    
    # Initialize GPT-2 model and tokenizer
    print("\nInitializing GPT-2 model...")
    model_name = 'gpt2'  # or 'gpt2-medium' for a larger model
    
    # Load tokenizer and add special tokens
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Load model with gradient checkpointing to save memory
    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        pad_token_id=tokenizer.pad_token_id
    ).to(config['device'])
    
    # Resize token embeddings and add special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Set generation parameters
    generation_config = {
        'max_length': 150,          # Maximum length of generated text
        'min_length': 30,           # Minimum length of generated text
        'no_repeat_ngram_size': 3,  # Prevent repetition of n-grams
        'early_stopping': True,     # Stop when all beam hypotheses reach EOS
        'num_beams': 4,             # Number of beams for beam search
        'temperature': 0.7,         # Lower for more focused, higher for more random
        'top_k': 50,                # Keep only top k tokens with highest probability
        'top_p': 0.95,              # Nucleus sampling: keep the top p% of probability mass
        'do_sample': True,          # Use sampling instead of greedy decoding
        'repetition_penalty': 1.2,  # Penalize repeated tokens
    }
    
    # Update model config with generation parameters
    for key, value in generation_config.items():
        setattr(model.config, key, value)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Using device: {config['device']}")
    
    # Optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': config['weight_decay'],
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=config['learning_rate'],
        eps=1e-8
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) // config['gradient_accumulation_steps'] * config['num_epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    
    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Print training info
    print(f"\nTraining for {config['num_epochs']} epochs")
    print(f"  Total optimization steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient accumulation steps: {config['gradient_accumulation_steps']}")
    print(f"  Total train batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Weight decay: {config['weight_decay']}")
    print(f"  Mixed precision training: {config['fp16']}")
    print(f"  Max sequence length: {config['max_length']}")
    print(f"  Number of training examples: {len(train_loader.dataset)}")
    print(f"  Number of validation examples: {len(val_loader.dataset)}")
    print(f"  Number of parameters: {num_params:,}")
    print(f"  Device: {config['device']}")
    print(f"  Save dir: {os.path.abspath(config['save_dir'])}")
    print()
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(enabled=config['fp16'])
    
    # Training loop
    best_val_loss = float('inf')
    best_rouge = 0.0
    global_step = 0
    
    # For tracking training statistics
    training_stats = []
    
    # Create output directory if it doesn't exist
    os.makedirs(config['save_dir'], exist_ok=True)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config['device'],
            config=config,
            scaler=scaler
        )
        
        # Evaluate on validation set
        val_loss, rouge_scores = evaluate(model, val_loader, config['device'], tokenizer, max_examples=2)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Print ROUGE scores if available
        if rouge_scores:
            print("\nAverage ROUGE Scores:")
            # Group ROUGE scores by metric type
            metrics = {}
            for key in sorted(rouge_scores.keys()):
                metric_name = key.split('_')[0]
                metric_type = '_'.join(key.split('_')[1:])  # precision, recall, or f1
                if metric_name not in metrics:
                    metrics[metric_name] = {}
                metrics[metric_name][metric_type] = rouge_scores[key]
            
            # Print formatted ROUGE scores
            for metric_name, scores in metrics.items():
                print(f"  {metric_name.upper()}:")
                for score_type, value in scores.items():
                    print(f"    {score_type}: {value:.4f}", end="  ")
                print()  # New line after each metric
        
        # Prepare epoch statistics
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        # Add ROUGE scores if available
        if rouge_scores:
            epoch_stats.update(rouge_scores)
        
        # Add to training stats
        training_stats.append(epoch_stats)
        
        # Save training statistics
        stats_file = os.path.join(config['save_dir'], 'training_stats.json')
        save_training_stats(training_stats, stats_file)
        
        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_dir = os.path.join(config['save_dir'], 'best_model')
            
            # Save model and tokenizer
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Save training arguments
            torch.save(config, os.path.join(output_dir, 'training_args.bin'))
            
            print(f"\nSaved best model to {output_dir} (val_loss: {val_loss:.4f})")
            
            # Also save the best model's training stats
            best_stats_file = os.path.join(output_dir, 'training_stats.json')
            save_training_stats(training_stats, best_stats_file)
        
        # Save checkpoint every save_interval epochs
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_dir = os.path.join(config['save_dir'], f'checkpoint-{epoch+1}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            
            # Save training arguments
            torch.save(config, os.path.join(checkpoint_dir, 'training_args.bin'))
            
            print(f"Saved checkpoint to {checkpoint_dir}")
        
        # Example generations are now handled in the evaluate function
        # No need for separate generation code here

if __name__ == "__main__":
    main()
