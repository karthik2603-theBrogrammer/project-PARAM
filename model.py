import torch
import torch.nn as nn
from transformer_block import TransformerBlock
from layer_norm import LayerNorm

class ParamModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout_emb = nn.Dropout(cfg['dropout_rate'])

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(cfg= cfg) for _ in range(cfg["num_layers"])
        ])
        self.final_norm = LayerNorm(emb_dim= cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias= False)

    def forward(self, inp):
        b, num_tokens = inp.shape
        token_emb = self.token_emb(inp)
        position_emb = self.position_emb(torch.arange(num_tokens, device= inp.device))
        
        x = token_emb + position_emb
        x = self.dropout_emb(x)
        for module in self.transformer_blocks:
            x = module(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


PARAM_CONFIG_1B = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 1536,
    "num_heads": 24,
    "num_layers": 36,
    "dropout_rate": 0.25,
    "qkv_bias": False
}

PARAM_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768, # Embedding dimension
    "num_heads": 12, # Number of attention heads
    "num_layers": 12, # Number of layers
    "dropout_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}
