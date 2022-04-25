import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from zmq import device
from .layers import *

# Attention Layers
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, key, value, query, mask=None):
        # [BS*H, L, N/H]
        qk = torch.bmm(query, key.transpose(1,2))
        qk = qk / math.sqrt(key.shape[2])
        if mask is not None:
            qk = qk.masked_fill(mask, -np.inf) 
        attn = F.softmax(qk ,dim=-1)
        qkv = torch.bmm(attn, value)
        return qkv, attn



class MultiHeadedAttention(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention()
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv


        # Masks
        if mask is not None:
            max_len = k.shape[1]
            self_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            self_attn_mask = self_attn_mask.repeat(self.n_head, 1, 1)
        else: self_attn_mask = None
        
        output, attn = self.attention(q, k, v, mask=self_attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn    
    
    
class FFTBlock(nn.Module):
    def __init__(self, in_d, hid_d, out_d,
                 n_heads, kernel=5, dropout=0.2):
        super(FFTBlock, self).__init__()
        
        self.n_heads = n_heads
        self.attention = MultiHeadedAttention(n_heads, in_d, in_d, in_d)
        self.post_layers = nn.Sequential(
            ConvLayer(in_d, hid_d, kernel),
            nn.LayerNorm(hid_d),
            nn.ReLU(),
            ConvLayer(hid_d, out_d, kernel),
            nn.LayerNorm(out_d),
            nn.Dropout(dropout))    
        
    def apply_pad_mask(self, x, mask=None):
        if mask is not None:
            return x.masked_fill(mask.unsqueeze(-1), 0)
        else: return x

    def forward(self, x, mask=None):
        # [BS, L, N]
        out, attention_score = self.attention(x, x, x, mask)
        out = self.apply_pad_mask(out, mask)
        residual = out.clone()
        out = self.post_layers(out) + residual
        out = self.apply_pad_mask(out, mask)
        return out, attention_score