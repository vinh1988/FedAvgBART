import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartConfig

class DistilBARTGen(nn.Module):
    def __init__(self, num_encoder_layers=6, num_decoder_layers=6, d_model=768, 
                 nhead=12, dim_feedforward=3072, dropout=0.1, 
                 max_length=1024, vocab_size=50265, use_pt_model=True):
        super(DistilBARTGen, self).__init__()
        
        self.use_pt_model = use_pt_model
        
        if use_pt_model:  # Use pre-trained weights
            self.model = BartForConditionalGeneration.from_pretrained(
                'facebook/bart-base',
                output_attentions=False,
                output_hidden_states=False,
            )
            self.config = self.model.config
        else:  # Initialize from scratch
            config = BartConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                encoder_layers=num_encoder_layers,
                decoder_layers=num_decoder_layers,
                encoder_attention_heads=nhead,
                decoder_attention_heads=nhead,
                encoder_ffn_dim=dim_feedforward,
                decoder_ffn_dim=dim_feedforward,
                max_position_embeddings=max_length,
                dropout=dropout,
                attention_dropout=dropout,
                activation_dropout=dropout,
                scale_embedding=True,
                num_labels=vocab_size,
                pad_token_id=1,  # BART uses 1 as padding token
                bos_token_id=0,
                eos_token_id=2,
                is_encoder_decoder=True,
                decoder_start_token_id=2,
                forced_eos_token_id=2,
            )
            self.model = BartForConditionalGeneration(config)
            self.config = config
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        return outputs
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def get_encoder(self):
        return self.model.get_encoder()
    
    def get_decoder(self):
        return self.model.get_decoder()
    
    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs
        )
        
        instance = cls(use_pt_model=True)
        instance.model = model
        instance.config = model.config
        return instance
