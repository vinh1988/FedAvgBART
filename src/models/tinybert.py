import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TinyBERT(nn.Module):
    def __init__(self, num_classes=4, pretrained_model_name='huawei-noah/TinyBERT_General_4L_312D', dropout=0.1, **kwargs):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=num_classes)
        self.bert = AutoModel.from_pretrained(pretrained_model_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1] if len(outputs) > 1 else outputs[0][:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits 