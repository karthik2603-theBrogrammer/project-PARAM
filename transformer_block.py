import torch
import torch.nn as nn
from typing import Optional
from feed_forward_network import FeedForwardNetwork
from moe import MOELayer
from layer_norm import LayerNorm
from mha import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg, use_moe = False) -> None:
        super().__init__()
        self.norm1 = LayerNorm(emb_dim= cfg["emb_dim"])
        self.mhaunit = MultiHeadAttention(
            d_in= cfg["emb_dim"],
            d_out= cfg["emb_dim"],
            context_length= cfg["context_length"],
            dropout= cfg["dropout_rate"],
            num_heads= cfg["num_heads"],
            qkv_bias= cfg["qkv_bias"]
        )
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.norm2 = LayerNorm(emb_dim= cfg["emb_dim"])
        self.moe = MOELayer(config= cfg) if use_moe else FeedForwardNetwork(cfg= cfg)
        # Use the Same Dropout for post ffn pass.

    def forward(self, x, attention_mask: Optional[torch.tensor]):
        shortcut = x
        x = self.norm1(x)
        x = self.mhaunit(x, attention_mask)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.moe(x)
        x = self.dropout(x)
        x = x + shortcut
        return x


