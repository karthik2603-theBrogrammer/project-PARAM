import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.epsilon = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim= -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        x_norm = (x - mean)/ torch.sqrt(var + self.epsilon)
        return self.scale * x_norm + self.shift




