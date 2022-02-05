import torch
import torch.nn as nn

class FastSpeech2Loss(nn.Module):
    def __init__(self, energy_feature_level, pitch_feature_level):
        super(FastSpeech2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.energy_feature_level = energy_feature_level
        self.pitch_feature_level = pitch_feature_level
        
        assert energy_feature_level in ['phoneme_frame', 'mel_frame'], \
            f'Incorrect energy feature level option, got: {energy_feature_level}'
        assert pitch_feature_level in ['phoneme_frame', 'mel_frame'], \
            f'Incorrect pitch feature level option, got: {pitch_feature_level}'
        
    def forward(self, predictions, targets, text_mask, mel_mask):
        mel_postnet, mel_preds, pitch_preds, energy_preds, log_duration_preds = predictions
        mel_targets, pitch_targets, energy_targets, duration_targets = targets 

        mel_targets = mel_targets.transpose(1,2)
        log_duration_targets = torch.log(duration_targets.float() + 1)

        # Apply masks
        text_mask = ~text_mask
        mel_mask = ~mel_mask
        mel_targets = mel_targets[:, : mel_mask.shape[1], :]
        mel_mask = mel_mask[:, :mel_mask.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_frame":
            pitch_preds = pitch_preds.masked_select(text_mask)
            pitch_targets = pitch_targets.masked_select(text_mask)
        elif self.pitch_feature_level == "mel_frame":
            pitch_preds = pitch_preds.masked_select(mel_mask)
            pitch_targets = pitch_targets.masked_select(mel_mask)

        if self.energy_feature_level == "phoneme_frame":
            energy_preds = energy_preds.masked_select(text_mask)
            energy_targets = energy_targets.masked_select(text_mask)
        elif self.energy_feature_level == "mel_frame":
            energy_preds = energy_preds.masked_select(mel_mask)
            energy_targets = energy_targets.masked_select(mel_mask)

        log_duration_preds = log_duration_preds.masked_select(text_mask)
        log_duration_targets = log_duration_targets.masked_select(text_mask)

        mel_preds = mel_preds.masked_select(mel_mask.unsqueeze(-1))
        mel_postnet = mel_postnet.masked_select(
            mel_mask.unsqueeze(-1)
        )
        
        if mel_targets.shape[1] == 80: 
            mel_targets = mel_targets.transpose(1,2)
             
        mel_targets = mel_targets.masked_select(mel_mask.unsqueeze(-1))
        
        mel_postnet_loss = self.mae_loss(mel_postnet, mel_targets.float())
        mel_loss = self.mae_loss(mel_preds, mel_targets.float())
        pitch_loss = self.mse_loss(pitch_preds, pitch_targets.float())
        energy_loss = self.mse_loss(energy_preds, energy_targets.float())
        
        durations_preds = torch.clamp(
                (torch.round(torch.exp(log_duration_preds) - 1) * 1), min=0)
        durations_targets = torch.clamp(
                (torch.round(torch.exp(log_duration_preds) - 1) * 1), min=0)
        
        duration_loss = self.mse_loss(log_duration_preds, log_duration_targets)
        
        print(f'''Mel Postnet loss: {mel_postnet_loss} 
                  Mel loss: {mel_loss}
                  Pitch Loss {pitch_loss}
                  Energy Loss {energy_loss}
                  Duration loss {duration_loss}''')
        
        return [mel_postnet_loss, mel_loss, pitch_loss, energy_loss, duration_loss]
    
