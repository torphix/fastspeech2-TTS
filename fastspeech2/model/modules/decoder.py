import torch.nn as nn

from .layers import CosinePositionalEncoding
from .transformer import FFTBlock



class Decoder(nn.Module):
    def __init__(self, in_d, hid_d, out_d,
                 n_blocks=6, n_heads=8, k_size=3):
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList()
        self.positional_enc = CosinePositionalEncoding(in_d)
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
    
    

    
