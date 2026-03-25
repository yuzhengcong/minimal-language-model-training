import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # W shape: (num_embeddings, embedding_dim)
        W = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(W)
        self.W = nn.Parameter(W)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 查表：用 token id 取出对应行
        return self.W[token_ids]
