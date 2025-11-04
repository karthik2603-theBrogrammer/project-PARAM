import torch
import torch.nn as nn
class MaskedSelfAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, qkv_bias: bool = True) -> None:
        super().__init__()
        self.w_query = nn.Linear(d_in, d_out, bias= qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias= qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias= qkv_bias)
        self.dropout = nn.Dropout(p= dropout)
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal= 1)
        )
    
    def forward(self, x):
        b_size, num_tokens, hidden_dim = x.shape # Batch Size, Number of Tokens, hidden dimension of the LLM.
        # ((1, 1, 1, 1, 1)
        # (1, 1, 1, 1, 1)   b_size = 4, num_tokens = 5, in z_axis there is hidden_dim.
        # (1, 1, 1, 1, 1)   b_size(batch size) refers to the number of sentences/ sequences in the batch.
        # (1, 1, 1, 1, 1))
        queries = self.w_query(x) 
        keys = self.w_key(x)
        values = self.w_value(x)

        attention_scores = queries @ keys.transpose(1, 2) # Exchange num_tokens and hidden_dim for matrix multiplicaiton purposes.
        attention_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attention_weights = nn.Softmax(attention_scores/ torch.sqrt(hidden_dim), dim= -1)
        attention_weights = self.dropout(attention_weights)
        context_vector = attention_weights * values
        return context_vector