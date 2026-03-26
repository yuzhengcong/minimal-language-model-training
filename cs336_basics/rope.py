import torch
from torch import nn


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k

        i = torch.arange(0, d_k // 2, dtype=torch.float32, device=device)
        freq = theta ** (-2 * i / d_k)

        positions = torch.arange(0, max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, freq)

        self.register_buffer("cos", torch.cos(angles))  # (max_seq_len, d_k/2)
        self.register_buffer("sin", torch.sin(angles))  # (max_seq_len, d_k/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape: (..., seq_len, d_k)
        # token_positions shape: (..., seq_len)

        # 用 token_positions 取出对应位置的 cos/sin，shape: (..., seq_len, d_k/2)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        new_even = x_even * cos - x_odd * sin
        new_odd  = x_even * sin + x_odd * cos

        return torch.stack([new_even, new_odd], dim=-1).reshape(x.shape)
