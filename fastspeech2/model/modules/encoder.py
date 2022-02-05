import torch.nn as nn
from .transformer import FFTBlock
from .layers import PositionalEncoding

# Encoder  
class Encoder(nn.Module):
    def __init__(self, in_d, hid_d, out_d,
                 n_blocks=6, n_heads=8, k_size=3):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.positional_enc = PositionalEncoding(in_d)
        
        for n in range(n_blocks):
            self.layers.append(
                    FFTBlock(
                        in_d,
                        hid_d, 
                        out_d,
                        n_heads,
                        k_size))
        
    def forward(self, x, mask=None):
        x = self.positional_enc(x)
        for layer in self.layers:
            x, _ = layer(x, mask)
        return x