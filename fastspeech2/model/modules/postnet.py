import torch
import torch.nn as nn
from .layers import ConvLayer

class Postnet(nn.Module):
    def __init__(self, n_layers, dropout=0.2):
        super(Postnet, self).__init__()

        self.layers = nn.ModuleList([
            ConvLayer(80, 256, kernel_size=5),
            nn.Tanh(),
            nn.Dropout(dropout)])
        
        for n in range(n_layers-2):
            self.layers.append(nn.Sequential(
                ConvLayer(256, 256, kernel_size=5),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.LayerNorm(256),         
            ))
            
        self.layers.append(
           nn.Sequential(
            ConvLayer(256, 80, kernel_size=5),
           ))
        
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)       