import enum
import json
import torch
import math
from ...data.utils import pad
import torch.nn as nn
import torch.nn.functional as F
from .layers import ConvLayer
from torch import Tensor


@torch.jit.script
def get_item(x: Tensor):
    item = x.int()
    return item


class LengthRegulator(nn.Module):
    '''
    Inputs:
        - Phonemes: [BS, L, N]
        - Durations: [BS, eL] 
        eL values = int specifying how much to expand by, already alpha scaled
        eL values should equal L
        - max_len: mels get cropped to this value
    Ouptuts:
        - Phonemes: [BS, L*eL, N]
        - mel_lens: [BS] BS = length of each mel without padding
    '''
    def __init__(self, onnx_export=False):
        super(LengthRegulator, self).__init__()
        
    def forward(self, x, durations, duration_alpha=1.0):
        # Preprocess durations
        if isinstance(durations[0][0].item(), int):
            durations = torch.clamp(
                    (torch.round((durations*duration_alpha).float())), min=0)
        else: # Inference, convert from log
            durations = torch.clamp(
                (torch.round(torch.exp(durations) - 1) * duration_alpha), min=0)
        # Expand
        mel_lens, expanded = [], []
        for idx, batch in enumerate(x):
            duration = durations[idx]
            expanded.append(torch.repeat_interleave(batch, duration.long(), dim=0))
            mel_lens.append(torch.sum(duration))
        expanded = pad(expanded)
        mel_lens = torch.tensor(mel_lens)
        return expanded, mel_lens
            

class Predictor(nn.Module):
    def __init__(self, in_d, hid_d, out_d, k_size, dropout=0.5):
        super(Predictor, self).__init__()

        self.layers = nn.Sequential(
            # L1
            ConvLayer(in_d, hid_d, k_size),
            nn.ReLU(),
            nn.LayerNorm(hid_d),
            nn.Dropout(dropout),
            # # L2
            ConvLayer(hid_d, hid_d, k_size),
            nn.ReLU(),
            # nn.LayerNorm(hid_d),
            nn.Dropout(dropout),
            # L3
            ConvLayer(hid_d, out_d, k_size),
            nn.ReLU(),
            nn.LayerNorm(out_d),
            nn.Dropout(dropout),
            # Projection
            nn.Linear(out_d, 1))
    
    def forward(self, x, mask=None):
        x = self.layers(x)
        x = x.squeeze(-1)
        if mask is not None:
            x = x.masked_fill(mask, 0.0)
        return x
    
        
class VarianceAdaptor(nn.Module):
    def __init__(self, variance_config, audio_metadata):
        super(VarianceAdaptor, self).__init__()
        # Config
        in_d = variance_config['in_d']
        hid_d = variance_config['hid_d']
        out_d = variance_config['out_d']
        k_size = variance_config['k_size']
        quantization_type = variance_config['quantization_type']
        n_bins = variance_config['n_bins']
        # Feature level sets the dim for energy and pitch
        self.pitch_feature_level = variance_config['pitch_feature_level']
        self.energy_feature_level = variance_config['energy_feature_level']
        assert quantization_type in ['linear', 'log'], \
            'Quantization type must be either linear or log'
        assert self.pitch_feature_level in ['mel_frame', 'phoneme_frame'], \
            'Pitch feature level must == mel_frame or phoneme_frame'
        assert self.energy_feature_level in ['mel_frame', 'phoneme_frame'], \
            'Energy feature level must == mel_frame or phoneme_frame'
        # Layers
        self.duration_predictor = Predictor(in_d, hid_d, out_d, k_size)
        self.pitch_predictor = Predictor(in_d, hid_d, out_d, k_size)
        self.energy_predictor = Predictor(in_d, hid_d, out_d, k_size)
        self.length_regulator = LengthRegulator(variance_config['onnx_export'])
        
        # Quantization
        pitch_min, pitch_max = audio_metadata['pitch']['min'], audio_metadata['pitch']['max']
        energy_min, energy_max = audio_metadata['energy']['min'], audio_metadata['energy']['max']
        # Pitch & Energy embeddings
        self.pitch_bins, self.pitch_embedding = self.create_embeddings(quantization_type, pitch_min, 
                                                                        pitch_max, n_bins, in_d)
        self.energy_bins, self.energy_embedding = self.create_embeddings(quantization_type, energy_min, 
                                                                        energy_max, n_bins, in_d)
        
    def create_embeddings(self, quantization_type, 
                          min_v, max_v, n_bins, emb_in_d):
        if quantization_type == 'log':
            bins = nn.Parameter(
                torch.exp(torch.linspace(torch.log(min_v), torch.log(max_v), n_bins -1)),
                requires_grad=False)
        elif quantization_type == 'linear': 
            bins = nn.Parameter(
                torch.linspace(min_v, max_v, n_bins-1), 
                requires_grad=False)
        embedding = nn.Embedding(n_bins, emb_in_d)
        return bins, embedding
    
    def bucketize(self, tensor, bucket_boundaries):
        result = torch.zeros_like(tensor, dtype=torch.int32)
        for boundary in bucket_boundaries:
            result += (tensor > boundary).int()
        return result.long()
        
    def embed_pitch(self, x, ground_truth, alpha, mask=None):
        pitch_preds = self.pitch_predictor(x, mask)
        if ground_truth is None: # Inference
            pitch_preds = pitch_preds * alpha
            embedding = self.pitch_embedding(
                self.bucketize(pitch_preds, self.pitch_bins))
        else: # Training
            embedding = self.pitch_embedding(
                self.bucketize(ground_truth, self.pitch_bins))
        return pitch_preds, embedding

    def embed_energy(self, x, ground_truth, alpha, mask=None):
        energy_preds = self.energy_predictor(x, mask)
        if ground_truth is None: # Inference
            energy_preds = energy_preds * alpha
            embedding = self.energy_embedding(
                self.bucketize(energy_preds, self.energy_bins))
        else: # Training
            embedding = self.energy_embedding(
                self.bucketize(ground_truth, self.energy_bins))
        return energy_preds, embedding
    
    def forward(self, phonemes, 
                phoneme_masks=None,
                mel_masks=None,
                true_pitch=None, 
                true_energy=None, 
                durations=None,
                pitch_alpha=1.0,
                energy_alpha=1.0,
                duration_alpha=1.0,
                ):
        '''
        Phonemes: [BS, L, N]
        Durations: [BS, L]
        Phoneme masks: Blank out padded sequences
        Alpha: Control energy, pitch, duration
        True: Ground truth values for training
        '''
        x = phonemes
        log_durations_pred = self.duration_predictor(phonemes, phoneme_masks)
        # Inference
        if durations is None: durations = log_durations_pred
            
        # Phoneme Frame
        if self.pitch_feature_level == 'phoneme_frame':
            pitch_preds, pitch_embedding = self.embed_pitch(
                x, true_pitch, pitch_alpha, phoneme_masks)
            x = x + pitch_embedding
            
        if self.energy_feature_level == 'phoneme_frame':
            energy_preds, energy_embedding = self.embed_energy(
                x, true_energy, energy_alpha, phoneme_masks)  
            x = x + energy_embedding
            
        # Expand
        x, mel_len = self.length_regulator(x, durations, duration_alpha)

        # Mel Frame
        if self.pitch_feature_level == 'mel_frame':
            pitch_preds, pitch_embedding = self.embed_pitch(
                x, true_pitch, pitch_alpha, mel_masks)
            
        if self.energy_feature_level == 'mel_frame':
            energy_preds, energy_embedding = self.embed_energy(
                x, true_energy, energy_alpha, mel_masks)        
        
        return x, pitch_preds, energy_preds, log_durations_pred
        
