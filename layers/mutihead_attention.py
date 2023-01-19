import torch
from torch import Tensor, nn
import torch.nn.functional as f
from utils import scaled_dot_product_attention

class AttentionHead(nn.Module):
    def __init__(self, dim_in:int, dim_q:int, dim_k:int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))
    
class MultiheadAttention(nn.Module):
    def __init__(self, num_heads:int,  dim_in:int, dim_q:int, dim_k:int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads*dim_k, dim_in)
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query,key,value) for h in self.heads], dim=-1)
        )
