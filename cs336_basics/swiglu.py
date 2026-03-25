import torch
from torch import nn
from cs336_basics.linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # W1, W3: 升维 d_model -> d_ff
        # W2: 降维 d_ff -> d_model
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(W1·x) ⊙ (W3·x)，再经 W2 投影
        w1x = self.w1(x)
        silu = w1x * torch.sigmoid(w1x)   # SiLU = x * sigmoid(x)
        return self.w2(silu * self.w3(x))
