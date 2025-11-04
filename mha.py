import torch
import torch.nn as nn
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 d_in: int, 
                 d_out: int, 
                 context_length: int, 
                 dropout: float, 
                 num_heads:int, 
                 qkv_bias: bool = False
    ) -> None:
        super().__init__()
        # d_in: Input embedding dimension of the LLM.
        # d_out: Output embedding dimension of the LLM, must be divisible by num_heads.

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_size = int(d_out/ num_heads)
        self.W_query = nn.Linear(d_in, d_out, bias= qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias= qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias= qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)                 #
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal= 1)
        )

    def forward(self, x, attention_mask: Optional[torch.tensor]):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_size)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_size)
        values = values.view(b, num_tokens, self.num_heads, self.head_size)

        queries = queries.transpose(1, 2)  # Change from (b, num_tokens, num_heads, head_size) to (b, num_heads, num_tokens, head_size)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        mask_bool = mask_bool.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
        attn_scores = attn_scores/ keys.shape[-1] ** 0.5

        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            attn_scores = attn_scores + attention_mask
            
        attn_weights = torch.softmax(attn_scores, dim = -1)
        attn_weights = self.dropout(attn_weights)

        context_vector = (attn_weights @ values).transpose(1, 2) # Back to (b, num_tokens, num_heads, head_size)
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        context_vector = self.out_proj(context_vector)

        return context_vector





