import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

class Range:
    """Range class for argument parsing"""
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __repr__(self):
        return f"[{self.start}, {self.end}]"

class TensorBoardRunner:
    """Helper class to run TensorBoard"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def run(self, port=6006):
        from tensorboard import program
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_dir, '--port', str(port)])
        url = tb.launch()
        print(f"TensorBoard started at {url}")
        return url

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_args(args):
    """Check command line arguments"""
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory {args.data_dir} does not exist")
    os.makedirs(args.output_dir, exist_ok=True)
    return args

def init_weights(module):
    """Initialize weights for neural network modules"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

class TqdmToLogger(object):
    """Output stream for TQDM to log to logger"""
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buffer = ''

    def write(self, buf):
        if buf.endswith('\n'):
            self.buffer += buf.rstrip('\n')
            self.logger.log(self.level, self.buffer)
            self.buffer = ''
        else:
            self.buffer += buf

    def flush(self):
        if len(self.buffer) > 0:
            self.logger.log(self.level, self.buffer)
            self.buffer = ''

class MetricManager:
    """Class to manage training metrics"""
    def __init__(self):
        self.metrics = {}
        
    def update(self, metrics_dict):
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_metric(self, metric_name):
        return self.metrics.get(metric_name, [])
    
    def get_latest(self, metric_name):
        values = self.metrics.get(metric_name, [])
        return values[-1] if values else None
    
    def get_average(self, metric_name):
        values = self.metrics.get(metric_name, [])
        return sum(values) / len(values) if values else 0.0

def stratified_split(dataset, labels, test_size=0.2, random_state=42):
    """Split dataset into train and test sets with stratification"""
    from sklearn.model_selection import train_test_split
    return train_test_split(
        range(len(dataset)), 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels,
        shuffle=True
    )
