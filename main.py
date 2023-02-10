import torch
from torch import Tensor, nn
import torch.nn.functional as f
from layers.encoder import TransformerEncoder
from layers.decoder import TransformerDecoder
    
class Transformer(nn.Module):
    def __init__(
            self,
            num_encoder_layers:int = 6,
            num_decoder_layers:int = 6,
            dim_model: int = 512, 
            num_heads: int=6, 
            dim_feedforward: int=2048, 
            dropout:float=0.1,
            activation: nn.Module = nn.ReLU()
            ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
    def forward(self, src:Tensor, tgt:Tensor) -> Tensor:
        return self.decoder(tgt, self.encoder(src))
    
## Testing
device = 'cuda' if torch.cuda.is_available() else 'cpu'
src = torch.rand(64, 32, 512).to(device)
tgt = torch.rand(64, 16, 512).to(device)
out = Transformer().to(device)(src,tgt)
print(out.shape)