import torch
import torch.nn as nn
from .attention import DecayMaskedMultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = DecayMaskedMultiHeadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, decaymask=None):
        x = x+self.attn(self.norm1(x), decaymask)
        x = x+self.ffn(self.norm2(x))
        return x
    
class MultiModalTransformer(nn.Module):
    def __init__(self, dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, decaymask=None):
        for block in self.blocks:
            x = block(x, decaymask)
        return x
