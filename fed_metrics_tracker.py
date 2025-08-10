import os
import json
import time
import psutil
import numpy as np
from datetime import datetime
from collections import defaultdict
import rouge_score.rouge_scorer as rouge_scorer
import torch
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from rouge_score import rouge_scorer

class MetricsTracker:
    def __init__(self, output_dir, experiment_name, num_clients):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.num_clients = num_clients
        self.start_time = time.time()
        self.metrics = defaultdict(dict)
        self.client_metrics = defaultdict(dict)
        self.system_metrics = defaultdict(dict)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metrics_file = os.path.join(output_dir, f"{experiment_name}_metrics_{timestamp}.json")
        
    def log_round_metrics(self, round_num, metrics_dict, client_id=None):
        """Log metrics for a specific round and client"""
        if client_id is not None:
            if round_num not in self.client_metrics[client_id]:
                self.client_metrics[client_id][round_num] = {}
            self.client_metrics[client_id][round_num].update(metrics_dict)
        else:
            if round_num not in self.metrics:
                self.metrics[round_num] = {}
            self.metrics[round_num].update(metrics_dict)
        
        # Save metrics after each update
        self._save_metrics()
    
    def log_system_metrics(self):
        """Log system metrics (CPU, memory usage, etc.)"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        self.system_metrics[time.time()] = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_rss_mb': memory_info.rss / (1024 * 1024),
            'memory_percent': process.memory_percent(),
            'elapsed_time_seconds': time.time() - self.start_time
        }
    
    def calculate_bleu(self, predictions, references):
        """Calculate BLEU scores"""
        # Convert to format expected by NLTK
        refs = [[ref.split()] for ref in references]
        preds = [pred.split() for pred in predictions]
        
        # Calculate BLEU scores
        smoothie = SmoothingFunction().method4
        try:
            return {
                'bleu1': corpus_bleu(refs, preds, weights=(1, 0, 0, 0), smoothing_function=smoothie),
                'bleu2': corpus_bleu(refs, preds, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
                'bleu3': corpus_bleu(refs, preds, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie),
                'bleu4': corpus_bleu(refs, preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie),
            }
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
            return {f'bleu{i}': 0.0 for i in range(1, 5)}
    
    def calculate_meteor(self, predictions, references):
        """Calculate METEOR score"""
        try:
            scores = []
            for ref, pred in zip(references, predictions):
                try:
                    score = meteor_score([ref], pred)
                    scores.append(score)
                except Exception as e:
                    print(f"Error in METEOR calculation: {e}")
                    continue
            return np.mean(scores) if scores else 0.0
        except Exception as e:
            print(f"Error in METEOR calculation: {e}")
            return 0.0
    
    def calculate_rouge(self, predictions, references):
        """Calculate ROUGE scores"""
        try:
            # Initialize accumulators for each ROUGE metric
            rouge1_f1 = []
            rouge2_f1 = []
            rougeL_f1 = []
            
            for ref, pred in zip(references, predictions):
                try:
                    if not ref or not pred:
                        continue
                        
                    # Calculate ROUGE scores for this pair
                    scores = self.rouge_scorer.score(ref, pred)
                    
                    # Extract F1 scores for each ROUGE metric
                    rouge1_f1.append(scores['rouge1'].fmeasure)
                    rouge2_f1.append(scores['rouge2'].fmeasure)
                    rougeL_f1.append(scores['rougeL'].fmeasure)
                    
                except Exception as e:
                    print(f"Error in ROUGE calculation: {e}")
                    continue
            
            # Calculate averages, default to 0.0 if no valid scores
            num_scores = len(rouge1_f1)
            if num_scores == 0:
                return {
                    'rouge1_f1': 0.0,
                    'rouge2_f1': 0.0,
                    'rougeL_f1': 0.0
                }
            
            return {
                'rouge1_f1': float(sum(rouge1_f1) / num_scores),
                'rouge2_f1': float(sum(rouge2_f1) / num_scores),
                'rougeL_f1': float(sum(rougeL_f1) / num_scores)
            }
            
        except Exception as e:
            print(f"Error in ROUGE calculation: {e}")
            return {
                'rouge1_f1': 0.0,
                'rouge2_f1': 0.0,
                'rougeL_f1': 0.0
            }
    
    def calculate_bertscore(self, predictions, references):
        """Calculate BERTScore"""
        try:
            P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
            return {
                'bertscore_precision': P.mean().item(),
                'bertscore_recall': R.mean().item(),
                'bertscore_f1': F1.mean().item()
            }
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0
            }
    
    def calculate_all_metrics(self, predictions, references):
        """Calculate all text generation metrics"""
        if not predictions or not references:
            print("Warning: Empty predictions or references")
            return {}
            
        metrics = {}
        
        # Basic metrics
        metrics.update(self.calculate_bleu(predictions, references))
        metrics['meteor'] = self.calculate_meteor(predictions, references)
        
        # ROUGE metrics
        rouge_metrics = self.calculate_rouge(predictions, references)
        metrics.update(rouge_metrics)
        
        # BERTScore (more resource-intensive)
        if len(predictions) <= 100:  # Limit to prevent memory issues
            metrics.update(self.calculate_bertscore(predictions, references))
        
        return metrics
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                             np.int16, np.int32, np.int64, np.uint8,
                             np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, 
                               np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return super().default(obj)

    def _save_metrics(self):
        """Save metrics to disk"""
        # Convert all numpy types to native Python types before serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(x) for x in obj]
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                               np.int16, np.int32, np.int64, np.uint8,
                               np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, 
                               np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj

        metrics = {
            'global_metrics': convert_numpy_types(dict(self.metrics)),
            'client_metrics': {str(k): convert_numpy_types(dict(v)) 
                             for k, v in self.client_metrics.items()},
            'system_metrics': convert_numpy_types(dict(self.system_metrics)),
            'timestamp': datetime.now().isoformat(),
            'experiment_name': self.experiment_name,
            'num_clients': self.num_clients
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, cls=self.NumpyEncoder)
    
    def get_summary(self):
        """Generate a summary of all metrics"""
        return {
            'num_rounds': len(self.metrics),
            'num_clients': self.num_clients,
            'elapsed_time_seconds': time.time() - self.start_time,
            'metrics_file': self.metrics_file,
            'experiment_name': self.experiment_name
        }
