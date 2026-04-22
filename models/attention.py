import torch
import torch.nn as nn
import math

class DecayMaskedMultiHeadAttention(nn.Module):
    def __init__(self, dim, numheads):
        super().__init__()
        self.numheads = numheads
        self.dim = dim
        self.head_dim = dim // numheads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, decaymask = None):
        B, N, D = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, N, self.numheads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.numheads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.numheads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if decaymask is not None:
            scores = scores * decaymask.unsqueeze(0)

        attn = torch.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)