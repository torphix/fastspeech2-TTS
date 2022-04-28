import math
import torch
import torch.nn as nn
from ..utils import pad

# Custom Layers
class CosinePositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [BS, L, N] -> [L, BS, N]
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1, 0, 2).contiguous()
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x.permute(1, 0, 2).contiguous()
    

class PortaSpeechPositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, phonemes, words, 
                word_boundries, word_durations):
        # Phoneme pos encoding
        for idx, phoneme in enumerate(phonemes):
            ph_idxs = torch.nonzero(
                         pad(torch.split(
                             phoneme, word_boundries[idx], dim=0)))
        # Word pos encoding
        


class ConvLayer(nn.Module):
    def __init__(self, 
                 in_d, 
                 out_d, 
                 kernel_size=3, 
                 stride=1,
                 padding='same',
                 dilation=1,
                 dropout=0.3):
        super().__init__()
        self.in_d = in_d
        
        if padding == 'same':
            padding = (kernel_size-1)//2
            
        self.layers = nn.Sequential(
            nn.Conv1d(in_d, 
                      out_d, 
                      kernel_size, 
                      stride, 
                      padding, 
                      dilation))
        
    def forward(self, x):
        # Expects [BS, N, L]
        x = x.contiguous().transpose(1,2)
        x = self.layers(x)
        return x.contiguous().transpose(1,2)