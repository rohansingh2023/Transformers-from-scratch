import torch
from torch import Tensor, nn
import torch.nn.functional as f
from utils import feed_forward, position_encoding
from layers.residual import Residual
from layers.mutihead_attention import MultiheadAttention

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model: int = 512, num_heads: int=6, dim_feedforward: int=2048, dropout:float=0.1):
        super().__init__()
        dim_q = dim_k = max(dim_model//num_heads,1)
        self.attention = Residual(
            MultiheadAttention(num_heads,dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout
        )
    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)

    
class TransformerEncoder(nn.Module):
    def __init__(self,num_layers:int = 6, dim_model: int = 512, num_heads: int=8, dim_feedforward: int=2048,   dropout:float=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward,dropout)
                for _ in range(num_layers)
            ]
        )
    def forward(self, src:Tensor) -> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src += position_encoding(seq_len,dimension)
        for layer in self.layers:
            src = layer(src)
        return src
