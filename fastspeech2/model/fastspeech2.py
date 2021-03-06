import torch
import torch.nn as nn


from .modules.postnet import Postnet
from .modules.decoder import Decoder
from .modules.adaptors import VarianceAdaptor
from .modules.encoder import Encoder
from .modules.layers import ConvLayer, CosinePositionalEncoding

class FastSpeech2(nn.Module):
    def __init__(self, model_config, audio_metadata):
        super(FastSpeech2, self).__init__()
        
        dropout = model_config['dropout']
        # Encoder
        self.encoder = Encoder(model_config['phoneme_dict_size'],
                               model_config['pad_idx'],
                               model_config['emb_d'],
                               model_config['encoder']['hid_d'], 
                               model_config['encoder']['out_d'],
                               model_config['encoder']['n_blocks'],
                               model_config['encoder']['n_heads'],
                               model_config['encoder']['k_size'])
        self.variance_adaptor = VarianceAdaptor(model_config['variance_config'],
                                                audio_metadata)
        self.projection = ConvLayer(model_config['variance_config']['out_d'],
                                    model_config['decoder']['in_d'],
                                    kernel_size=1)
        self.decoder = Decoder(model_config['decoder']['in_d'],
                               model_config['decoder']['hid_d'], 
                               model_config['decoder']['out_d'],
                               model_config['decoder']['n_blocks'],
                               model_config['decoder']['n_heads'],
                               model_config['decoder']['k_size'])
        
        self.postnet = Postnet(model_config['postnet']['n_layers'], dropout)
        
    def forward(self, 
                phonemes, 
                duration_alpha=1.0,
                pitch_alpha=1.0,
                energy_alpha=1.0,
                text_masks=None,
                mel_masks=None,
                true_pitch=None, 
                true_energy=None, 
                durations=None,
                speaker_embedding=None,
                ):
        '''
        Phonemes: [BS, L, N]
        '''      
        # Phoneme
        phoneme_emb = self.encoder(phonemes, text_masks)
        if speaker_embedding is not None:
            phoneme_emb += speaker_embedding
        # Variance
        mel_preds, pitch_preds, energy_preds, log_duration_preds = \
            self.variance_adaptor(
                phoneme_emb,
                text_masks,
                mel_masks,
                true_pitch, 
                true_energy, 
                durations,
                pitch_alpha,
                energy_alpha,
                duration_alpha)
            
        # Postnet
        mel_preds = self.projection(mel_preds)            
        mel_preds = self.decoder(mel_preds, mel_masks)
        mel_postnet_pred = self.postnet(mel_preds) 
        return mel_postnet_pred, mel_preds, \
                pitch_preds, energy_preds, log_duration_preds


