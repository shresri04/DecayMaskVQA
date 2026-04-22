import torch
import torch.nn as nn

class DecayMask(nn.Module):
    def __init__(self, num_heads, init_sp = 1.0, lamda_val = 0.5):
        super().__init__()
        self.lamda_val = lamda_val
        self.start_points = nn.Parameter(torch.ones((num_heads,), init_sp))

    def forward(self, x):
        num_heads = self.start_points.shape[0]
        N = x.shape[0]

        masks = []
        for h in range(num_heads):
            sp = self.start_points[h]
            relu_term = torch.relu(x-sp)
            mask = self.lamda_val** relu_term
            masks.append(mask)

        return torch.stack(masks)