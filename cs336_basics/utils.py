import torch
from torch import nn
import math


def softmax(i: int, x: torch.Tensor) -> torch.Tensor:
    # 减去最大值防止 exp 溢出，keepdim 保持维度方便广播
    x_max = x.max(dim=i, keepdim=True).values
    x_stable = x - x_max

    # 计算 exp，再除以每行的和
    exp_x = torch.exp(x_stable)
    return exp_x / exp_x.sum(dim=i, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = softmax(-1, scores)

    return attn_weights @ V



class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

    def forward(
        self,
        x: torch.Tensor,
        q_proj_weight: torch.Tensor,
        k_proj_weight: torch.Tensor,
        v_proj_weight: torch.Tensor,
        o_proj_weight: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        rope=None,
    ) -> torch.Tensor:
        *batch, seq_len, d_model = x.shape
        h = self.num_heads
        d_k = self.d_k

        Q = x @ q_proj_weight.transpose(-2, -1)
        K = x @ k_proj_weight.transpose(-2, -1)
        V = x @ v_proj_weight.transpose(-2, -1)

        Q = Q.reshape(*batch, seq_len, h, d_k).transpose(-3, -2)
        K = K.reshape(*batch, seq_len, h, d_k).transpose(-3, -2)
        V = V.reshape(*batch, seq_len, h, d_k).transpose(-3, -2)

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
                # 扩展到 (..., num_heads, seq_len)
                for _ in batch:
                    token_positions = token_positions.unsqueeze(0)
                token_positions = token_positions.unsqueeze(-2).expand(*batch, h, seq_len)
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        out = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(-3, -2).reshape(*batch, seq_len, d_model)

        # 输出投影
        return out @ o_proj_weight.transpose(-2, -1)
