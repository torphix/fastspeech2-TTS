import torch.nn as nn
from .transformer import FFTBlock
from .layers import CosinePositionalEncoding

# Encoder  
class Encoder(nn.Module):
    def __init__(self, phoneme_d, pad_idx, emb_d, hid_d, out_d,
                 n_blocks=6, n_heads=8, k_size=3):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.phoneme_embedding = nn.Embedding(phoneme_d, emb_d, pad_idx)
        self.positional_enc = CosinePositionalEncoding(emb_d)
        
        for n in range(n_blocks):
            self.layers.append(
                    FFTBlock(
                        emb_d,
                        hid_d, 
                        out_d,
                        n_heads,
                        k_size))
        
    def forward(self, x, mask=None):
        x = self.phoneme_embedding(x)
        x = self.positional_enc(x)
        for layer in self.layers:
            x, _ = layer(x, mask)
        return x