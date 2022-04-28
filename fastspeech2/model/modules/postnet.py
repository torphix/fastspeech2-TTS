import torch
import torch.nn as nn
from .layers import ConvLayer

class Postnet(nn.Module):
    def __init__(self, n_layers, dropout=0.2):
        super(Postnet, self).__init__()

        self.layers = nn.ModuleList([
            nn.Conv1d(80, 256, kernel_size=5, padding=(5-1)//2),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(dropout)])
        
        for n in range(n_layers-2):
            self.layers.append(nn.Sequential(
                nn.Conv1d(256 if n == 0 else 512, 512,
                          kernel_size=5,
                          padding=(5-1)//2),
                nn.BatchNorm1d(512),         
                nn.Tanh(),
                nn.Dropout(dropout),
            ))
            
        self.layers.append(
           nn.Sequential(
            nn.Conv1d(512, 80, kernel_size=5,
                      padding=(5-1)//2),
            nn.BatchNorm1d(80),
           ))
        
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        x = x.transpose(1,2).contiguous()
        x = self.layers(x)
        x = x.transpose(1,2).contiguous()
        return x