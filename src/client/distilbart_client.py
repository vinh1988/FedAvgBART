import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import os
import logging
import copy
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from client.base import BaseClient
from models.distilbart_utils import DistilBARTWrapper

logger = logging.getLogger(__name__)

class DistilBARTClient(BaseClient):
    """Client for federated learning with DistilBART model."""
    
    def __init__(self, client_id: int, args=None, **kwargs):
        """Initialize DistilBART client.
        
        Args:
            client_id: Unique identifier for the client
            args: Arguments containing model and training parameters
            **kwargs: Additional arguments for the base client
        """
        # Store args before calling parent init
        self.args = args if args is not None else {}
        
        # Determine device
        self.use_cpu = getattr(self.args, 'use_cpu', False)
        device = 'cpu' if self.use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Remove device from kwargs if it exists to avoid duplicate argument
        kwargs.pop('device', None)
        
        # Call parent init with remaining kwargs and device
        super().__init__(client_id, device=device, **kwargs)
        
        # Extract model parameters with defaults and ensure correct types
        model_name = getattr(self.args, 'model_name', 'facebook/bart-large-cnn')
        max_source_length = int(getattr(self.args, 'max_source_length', 1024))
        max_target_length = int(getattr(self.args, 'max_target_length', 128))
        learning_rate = float(getattr(self.args, 'learning_rate', 5e-5))
        weight_decay = float(getattr(self.args, 'weight_decay', 0.01))
        
        # Initialize DistilBART model
        self.model_wrapper = DistilBARTWrapper(
            model_name=model_name,
            max_length=max_source_length,
            max_target_length=max_target_length,
            device=self.device
        )
        
        # Set up optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model_wrapper.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=1, 
            gamma=0.9  # Reduce learning rate by 10% each epoch
        )
    
    def train(self, num_epochs: int = 1, batch_size: int = 8) -> Dict[str, float]:
        """Train the model on client data.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing training metrics
        """
        # Set model to training mode
        self.model_wrapper.model.train()
        
        # Create data loader
        train_loader = self.get_train_loader(batch_size=batch_size)
        
        # Training loop
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                metrics = self.model_wrapper.train_step(batch)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model_wrapper.model.parameters(), 
                    max_norm=self.args.max_grad_norm
                )
                
                # Update parameters
                self.optimizer.step()
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Update metrics
                loss = metrics['loss']
                epoch_loss += loss
                total_loss += loss
                num_batches += 1
                
                # Log progress
                if (batch_idx + 1) % self.args.log_interval == 0:
                    logger.info(
                        f"Client {self.client_id} - Epoch: {epoch + 1}/{num_epochs}, "
                        f"Batch: {batch_idx + 1}/{len(train_loader)}, "
                        f"Loss: {loss:.4f}"
                    )
            
            # Log epoch metrics
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(
                f"Client {self.client_id} - Epoch: {epoch + 1}/{num_epochs}, "
                f"Avg Loss: {avg_epoch_loss:.4f}"
            )
        
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Get model parameters
        model_params = self.get_parameters()
        
        return {
            'loss': avg_loss,
            'num_samples': len(self.train_dataset),
            'model_params': model_params
        }
    
    def evaluate(self, dataset_type: str = 'test', batch_size: int = 8) -> Dict[str, float]:
        """Evaluate the model on client data.
        
        Args:
            dataset_type: Type of dataset to evaluate on ('train', 'val', or 'test')
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Set model to evaluation mode
        self.model_wrapper.model.eval()
        
        # Get data loader with proper error handling for missing datasets
        try:
            logger.info(f"Client {self.client_id} - Attempting to get {dataset_type} dataset...")
            if dataset_type == 'train':
                logger.info(f"Client {self.client_id} - Train dataset: {self.train_dataset}")
                if self.train_dataset is None:
                    logger.warning(f"Client {self.client_id} has no training dataset")
                    return {'loss': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'bleu': 0.0}
                data_loader = self.get_train_loader(batch_size=batch_size)
                logger.info(f"Client {self.client_id} - Train data loader created with {len(data_loader)} batches")
            elif dataset_type == 'val':
                logger.info(f"Client {self.client_id} - Val dataset: {self.val_dataset}")
                if self.val_dataset is None:
                    logger.warning(f"Client {self.client_id} has no validation dataset")
                    # Debug: Check if val_datasets was passed to the client
                    logger.warning(f"Client {self.client_id} - val_dataset attribute: {hasattr(self, 'val_dataset')}")
                    logger.warning(f"Client {self.client_id} - val_dataset value: {getattr(self, 'val_dataset', 'Not set')}")
                    return {'loss': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'bleu': 0.0}
                data_loader = self.get_val_loader(batch_size=batch_size)
                logger.info(f"Client {self.client_id} - Val data loader created with {len(data_loader)} batches")
            else:  # 'test'
                logger.info(f"Client {self.client_id} - Test dataset: {self.test_dataset}")
                if self.test_dataset is None:
                    logger.warning(f"Client {self.client_id} has no test dataset")
                    return {'loss': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'bleu': 0.0}
                data_loader = self.get_test_loader(batch_size=batch_size)
                logger.info(f"Client {self.client_id} - Test data loader created with {len(data_loader)} batches")
            
            # Log dataset sample
            if dataset_type == 'val' and hasattr(self, 'val_dataset') and self.val_dataset is not None:
                try:
                    sample = next(iter(data_loader))
                    logger.info(f"Client {self.client_id} - Sample batch from {dataset_type} dataset:")
                    logger.info(f"  Input IDs shape: {sample['input_ids'].shape if 'input_ids' in sample else 'N/A'}")
                    logger.info(f"  Attention mask shape: {sample['attention_mask'].shape if 'attention_mask' in sample else 'N/A'}")
                    logger.info(f"  Labels shape: {sample['labels'].shape if 'labels' in sample else 'N/A'}")
                except Exception as e:
                    logger.warning(f"Error logging {dataset_type} dataset sample: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error getting {dataset_type} data loader: {str(e)}", exc_info=True)
            return {'loss': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'bleu': 0.0}
        
        # Evaluation loop
        total_loss = 0.0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Forward pass
                metrics = self.model_wrapper.evaluate(batch)
                
                # Update metrics
                total_loss += metrics['loss']
                all_predictions.extend(metrics['predictions'])
                all_references.extend(metrics['references'])
        
        # Calculate average loss
        avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        num_samples = len(all_references)
        
        # Calculate ROUGE and BLEU scores using the evaluate package
        try:
            import evaluate
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            
            # Ensure we have valid predictions and references
            if not all_predictions or not all_references:
                logger.warning(f"No valid predictions or references for evaluation. Predictions: {len(all_predictions)}, References: {len(all_references)}")
                return {
                    'loss': avg_loss,
                    'rouge1': 0.0,
                    'rouge2': 0.0,
                    'rougeL': 0.0,
                    'bleu1': 0.0,
                    'bleu2': 0.0,
                    'bleu3': 0.0,
                    'bleu4': 0.0,
                    'gini_coefficient': 0.0,
                    'mean_contribution': 0.0,
                    'min_contribution': 0.0,
                    'max_contribution': 0.0,
                    'cv_contribution': 0.0,
                    'num_samples': num_samples
                }
            
            # Debug logging
            logger.debug(f"Raw predictions sample: {all_predictions[0] if all_predictions else 'No predictions'}")
            logger.debug(f"Raw references sample: {all_references[0] if all_references else 'No references'}")
            
            # Convert predictions and references to the correct format
            # For ROUGE: predictions and references should be lists of strings
            # For BLEU: predictions should be a list of token lists, references should be a list of lists of token lists
            
            # Process predictions and references for ROUGE and BLEU
            predictions = []
            tokenized_predictions = []
            tokenized_references = []
            
            for pred, ref in zip(all_predictions, all_references):
                # Skip None values
                if pred is None or ref is None:
                    continue
                    
                # Convert to string if needed
                pred_str = str(pred).strip() if pred else ""
                ref_str = str(ref).strip() if ref else ""
                
                # For ROUGE
                predictions.append(pred_str)
                
                # For BLEU (tokenize into words)
                pred_tokens = pred_str.split()
                ref_tokens = ref_str.split()
                
                tokenized_predictions.append(pred_tokens)
                tokenized_references.append([ref_tokens])  # BLEU expects list of references for each prediction
                
            # Calculate ROUGE scores if we have predictions
            if predictions and all_references:
                try:
                    rouge = evaluate.load('rouge')
                    rouge_scores = rouge.compute(
                        predictions=predictions,
                        references=all_references,
                        rouge_types=['rouge1', 'rouge2', 'rougeL'],
                        use_aggregator=True,
                        use_stemmer=True
                    )
                    rouge1 = rouge_scores['rouge1'] * 100  # Convert to percentage
                    rouge2 = rouge_scores['rouge2'] * 100
                    rougeL = rouge_scores['rougeL'] * 100
                except Exception as e:
                    logger.warning(f"Error calculating ROUGE scores: {e}")
                    rouge1 = rouge2 = rougeL = 0.0
            else:
                rouge1 = rouge2 = rougeL = 0.0
            
            # Calculate BLEU scores with smoothing if we have tokenized data
            if tokenized_predictions and tokenized_references:
                try:
                    smoothie = SmoothingFunction().method4
                    bleu_scores = {
                        'bleu1': corpus_bleu(tokenized_references, tokenized_predictions, weights=(1, 0, 0, 0), smoothing_function=smoothie) * 100,
                        'bleu2': corpus_bleu(tokenized_references, tokenized_predictions, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie) * 100,
                        'bleu3': corpus_bleu(tokenized_references, tokenized_predictions, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie) * 100,
                        'bleu4': corpus_bleu(tokenized_references, tokenized_predictions, smoothing_function=smoothie) * 100
                    }
                except Exception as e:
                    logger.warning(f"Error calculating BLEU scores: {e}")
                    bleu_scores = {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}
            else:
                bleu_scores = {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}
            
            # Calculate contribution metrics (example implementation - adjust as needed)
            # In a real implementation, you would track contributions during training
            try:
                # For now, we'll use a simple equal distribution
                # In a real implementation, you would track actual contributions
                if num_samples > 0:
                    contributions = [1.0 / num_samples] * num_samples
                    mean_contribution = 1.0 / num_samples
                    min_contribution = 1.0 / num_samples
                    max_contribution = 1.0 / num_samples
                    gini_coefficient = 0.0  # Perfect equality
                    cv_contribution = 0.0   # No variation
                else:
                    mean_contribution = 0.0
                    min_contribution = 0.0
                    max_contribution = 0.0
                    gini_coefficient = 0.0
                    cv_contribution = 0.0
            except Exception as e:
                logger.warning(f"Error calculating contribution metrics: {e}")
                mean_contribution = 0.0
                min_contribution = 0.0
                max_contribution = 0.0
                gini_coefficient = 0.0
                cv_contribution = 0.0
            
            # Prepare metrics dictionary with all required fields
            metrics = {
                'loss': float(avg_loss) if not isinstance(avg_loss, torch.Tensor) else avg_loss.item(),
                'rouge1': float(rouge1) if not isinstance(rouge1, torch.Tensor) else rouge1.item(),
                'rouge2': float(rouge2) if not isinstance(rouge2, torch.Tensor) else rouge2.item(),
                'rougeL': float(rougeL) if not isinstance(rougeL, torch.Tensor) else rougeL.item(),
                'bleu1': float(bleu_scores['bleu1']) if not isinstance(bleu_scores['bleu1'], torch.Tensor) else bleu_scores['bleu1'].item(),
                'bleu2': float(bleu_scores['bleu2']) if not isinstance(bleu_scores['bleu2'], torch.Tensor) else bleu_scores['bleu2'].item(),
                'bleu3': float(bleu_scores['bleu3']) if not isinstance(bleu_scores['bleu3'], torch.Tensor) else bleu_scores['bleu3'].item(),
                'bleu4': float(bleu_scores['bleu4']) if not isinstance(bleu_scores['bleu4'], torch.Tensor) else bleu_scores['bleu4'].item(),
                'gini_coefficient': float(gini_coefficient),
                'mean_contribution': float(mean_contribution),
                'min_contribution': float(min_contribution),
                'max_contribution': float(max_contribution),
                'cv_contribution': float(cv_contribution),
                'num_samples': int(num_samples) if not isinstance(num_samples, torch.Tensor) else num_samples.item()
            }
            
            # Log metrics
            logger.info(
                f"Client {self.client_id} - {dataset_type.capitalize()} - "
                f"Loss: {metrics['loss']:.4f}, "
                f"ROUGE-1: {metrics['rouge1']:.2f}, "
                f"ROUGE-2: {metrics['rouge2']:.2f}, "
                f"ROUGE-L: {metrics['rougeL']:.2f}, "
                f"BLEU-4: {metrics['bleu4']:.2f}"
            )
            
            return metrics
            
        except ImportError as e:
            logger.warning(f"Required packages not found: {e}. Using dummy metrics.")
            return {
                'loss': avg_loss.item() if hasattr(avg_loss, 'item') else float(avg_loss),
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'bleu1': 0.0,
                'bleu2': 0.0,
                'bleu3': 0.0,
                'bleu4': 0.0,
                'gini_coefficient': 0.0,
                'mean_contribution': 0.0,
                'min_contribution': 0.0,
                'max_contribution': 0.0,
                'cv_contribution': 0.0,
                'num_samples': num_samples.item() if hasattr(num_samples, 'item') else int(num_samples)
            }
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            return {
                'loss': avg_loss.item() if hasattr(avg_loss, 'item') else float(avg_loss),
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'bleu1': 0.0,
                'bleu2': 0.0,
                'bleu3': 0.0,
                'bleu4': 0.0,
                'gini_coefficient': 0.0,
                'mean_contribution': 0.0,
                'min_contribution': 0.0,
                'max_contribution': 0.0,
                'cv_contribution': 0.0,
                'num_samples': num_samples.item() if hasattr(num_samples, 'item') else int(num_samples)
            }
                
            # Process predictions
            predictions = []
            for pred in all_predictions:
                if isinstance(pred, (list, tuple)):
                    # If it's a list of tokens, join them with spaces
                    pred = ' '.join(str(token) for token in pred if token is not None)
                if not isinstance(pred, str):
                    pred = str(pred) if pred is not None else ""
                predictions.append(pred.strip())
            
            # Process references
            references = []
            for ref in all_references:
                if isinstance(ref, (list, tuple)):
                    # If it's a list of tokens, join them with spaces
                    if all(isinstance(token, (str, int, float)) for token in ref):
                        ref = ' '.join(str(token) for token in ref if token is not None)
                    # If it's a list of lists (multiple references), take the first one
                    elif ref and isinstance(ref[0], (list, tuple)):
                        ref = ' '.join(str(token) for token in ref[0] if token is not None)
                if not isinstance(ref, str):
                    ref = str(ref) if ref is not None else ""
                references.append(ref.strip())
            
            # Ensure we have the same number of predictions and references
            min_len = min(len(predictions), len(references))
            if min_len == 0:
                logger.warning("No valid predictions or references after processing")
                return {
                    'loss': avg_loss,
                    'rouge1': 0.0,
                    'rouge2': 0.0,
                    'rougeL': 0.0,
                    'bleu': 0.0
                }
            
            predictions = predictions[:min_len]
            references = references[:min_len]
            
            # Debug logging
            logger.debug(f"Processed prediction sample: {predictions[0][:100]}..." if predictions else "No predictions")
            logger.debug(f"Processed reference sample: {references[0][:100]}..." if references else "No references")
            
            # Calculate ROUGE scores
            try:
                rouge = evaluate.load('rouge')
                rouge_scores = rouge.compute(
                    predictions=predictions,
                    references=references,
                    use_stemmer=True
                )
                rouge1 = rouge_scores['rouge1']
                rouge2 = rouge_scores['rouge2']
                rougeL = rouge_scores['rougeL']
            except Exception as e:
                logger.error(f"Error calculating ROUGE scores: {str(e)}")
                rouge1 = rouge2 = rougeL = 0.0
            
            # Calculate BLEU score
            try:
                import nltk
                nltk.download('punkt', quiet=True)
                from nltk.tokenize import word_tokenize
                
                # For BLEU, we need to ensure we have valid string inputs
                # and handle the case where references might be empty or invalid
                if not predictions or not references:
                    logger.warning("Skipping BLEU calculation due to empty predictions or references")
                    bleu_score = 0.0
                else:
                    # Ensure we have valid strings
                    valid_preds = []
                    valid_refs = []
                    
                    for pred, ref in zip(predictions, references):
                        if not isinstance(pred, str) or not isinstance(ref, str):
                            continue
                        if not pred.strip() or not ref.strip():
                            continue
                        valid_preds.append(pred)
                        valid_refs.append([ref])  # Wrap in a list for BLEU
                    
                    if not valid_preds or not valid_refs:
                        logger.warning("No valid predictions or references for BLEU calculation")
                        bleu_score = 0.0
                    else:
                        # Calculate BLEU using sacrebleu for better compatibility
                        try:
                            import sacrebleu
                            
                            # Convert to the format expected by sacrebleu
                            refs = [[r[0]] for r in valid_refs]  # List of lists of references
                            
                            # Calculate BLEU score
                            bleu_score = sacrebleu.corpus_bleu(
                                valid_preds,
                                refs,
                                tokenize='13a',  # Standard tokenization
                                lowercase=False
                            ).score / 100.0  # Convert from percentage to 0-1 range
                            
                            logger.debug(f"BLEU score: {bleu_score:.4f}")
                            
                        except ImportError:
                            logger.warning("sacrebleu not installed, using nltk BLEU (less accurate)")
                            # Fall back to nltk BLEU if sacrebleu is not available
                            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
                            
                            # Tokenize the texts
                            tokenized_preds = [word_tokenize(pred.lower()) for pred in valid_preds]
                            tokenized_refs = [[word_tokenize(ref[0].lower())] for ref in valid_refs]
                            
                            # Calculate BLEU with smoothing
                            smoothie = SmoothingFunction().method1
                            bleu_score = corpus_bleu(
                                tokenized_refs,
                                tokenized_preds,
                                smoothing_function=smoothie
                            )
                            
            except Exception as e:
                logger.error(f"Error calculating BLEU score: {str(e)}", exc_info=True)
                bleu_score = 0.0
            
            # Log metrics
            logger.info(f"ROUGE-1: {rouge1:.4f}, ROUGE-2: {rouge2:.4f}, ROUGE-L: {rougeL:.4f}, BLEU: {bleu_score:.4f}")
            
            return {
                'loss': avg_loss,
                'rouge1': rouge1,
                'rouge2': rouge2,
                'rougeL': rougeL,
                'bleu': bleu_score,
                'num_samples': num_samples
            }
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            # Return basic metrics if evaluation fails
            return {
                'loss': avg_loss,
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'bleu': 0.0,
                'num_samples': num_samples
            }
        bleu_scores = bleu.compute(
            predictions=[[p.split()] for p in all_predictions],
            references=[[r.split()] for r in all_references]
        )
        
        return {
            'loss': avg_loss,
            'rouge1': rouge_scores['rouge1'].mid.fmeasure,
            'num_samples': num_samples,
            'rouge2': rouge_scores['rouge2'].mid.fmeasure,
            'rougeL': rouge_scores['rougeL'].mid.fmeasure,
            'bleu': bleu_scores['bleu'],
            'num_samples': len(all_predictions)
        }
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters."""
        return self.model_wrapper.get_model_parameters()
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters."""
        self.model_wrapper.set_model_parameters(parameters)
    
    def save_model(self, output_dir: str):
        """Save model to directory."""
        os.makedirs(output_dir, exist_ok=True)
        self.model_wrapper.save_pretrained(output_dir)
    
    def load_model(self, model_path: str):
        """Load model from directory."""
        self.model_wrapper = DistilBARTWrapper.from_pretrained(model_path)
