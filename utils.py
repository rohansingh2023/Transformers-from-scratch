import torch
from torch import Tensor, nn
import torch.nn.functional as f

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1,2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp/scale, dim=-1)
    return softmax.bmm(value)

def position_encoding(seq_len:int, dim_model:int, device: torch.device = torch.device("cpu")) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1,-1,1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1,1,-1)
    phase = pos / (1e4 ** (dim // dim_model))
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def feed_forward(dim_input: int = 512, dim_feedForward:int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedForward),
        nn.ReLU(),
        nn.Linear(dim_feedForward,dim_input)
    )