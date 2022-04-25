import yaml
import torch
import pytorch_lightning as ptl
from .loss import FastSpeech2Loss
from .fastspeech2 import FastSpeech2
from .utils import get_mask_from_lengths, plot_mel
from torch.optim.lr_scheduler import ReduceLROnPlateau

class FS2TrainingModule(ptl.LightningModule):
    def __init__(self, 
                 model_config, 
                 audio_metadata,
                 train_dl=None, 
                 val_dl=None):
        super(FS2TrainingModule, self).__init__()
        # Config
        with open(model_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # Model
        self.fs2 = FastSpeech2(self.config['model'], audio_metadata)
        self.loss = FastSpeech2Loss(self.config['model']['variance_config']['energy_feature_level'],
                                    self.config['model']['variance_config']['pitch_feature_level'])
        
        # Data
        self.train_dl, self.val_dl = train_dl, val_dl

    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    def forward(self, 
                phonemes,
                durations,
                energy,
                pitch,
                text_masks,
                mel_masks,
                mels=None,
                mel_lens=None,
                speaker_emb=None):     
        
        return self.fs2(phonemes=phonemes,
                        text_masks=text_masks,
                        mel_masks=mel_masks,
                        true_pitch=pitch,
                        true_energy=energy,
                        durations=durations,
                        speaker_embedding=speaker_emb) 
    
    def training_step(self, batch, batch_idx):
        (mels, 
         text, 
         durations, 
         energies, 
         pitchs, 
         text_masks, 
         mel_masks,
         mel_lens,
         speakers) = self.unpack_batch(batch)
        print(torch.max(mels), torch.max(energies))
        output = self.forward(phonemes=text,
                              durations=durations,
                              energy=energies,
                              pitch=pitchs,
                              text_masks=text_masks,
                              mel_masks=mel_masks,
                              mels=mels,
                              mel_lens=mel_lens,
                              speaker_emb=speakers)
        test = output[0].clone().transpose(1,2).detach().cpu().numpy()
        print(batch['raw_text'])
        plot_mel(test, ['Ouptut'])
        
        loss = self.loss(output, 
                         (mels, pitchs, energies, durations),
                         text_masks, mel_masks)
        
        logs = {
            'total_loss':sum(loss),
            'mel_postnet_loss': loss[0],
            'mel_loss': loss[1],
            'pitch_loss': loss[2],
            'energy_loss': loss[3],
            'duration_loss': loss[4]
            }
        
        return {'loss': sum(loss), 'logs': logs}
    
    def validation_step(self, batch, batch_idx):
        (mels, 
         text, 
         durations, 
         energies, 
         pitchs, 
         text_masks, 
         mel_masks,
         mel_lens,
         speakers) = self.unpack_batch(batch)
        
        output = self.forward(text,
                              durations,
                              energies,
                              pitchs,
                              text_masks,
                              mel_masks,
                              mels,
                              mel_lens,
                              speaker_emb=speakers)

        loss = self.loss(output, 
                         (mels, pitchs, energies, durations),
                         text_masks, mel_masks)
        
        self.log('val_loss', sum(loss))
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {'optimizer':optimizer,
                'scheduler': ReduceLROnPlateau(optimizer)}
    
    def unpack_batch(self, batch):
        # Note this is bad practice as is slow, should init on device
        text_mask, mel_masks = self.get_masks(batch['text_lens'], batch['text_max_len'],
                                              batch['mel_lens'], batch['mel_max_len'])
        
        return (
            batch['mels'], 
            batch['phonemes'], 
            batch['durations'], 
            batch['energies'],
            batch['pitchs'], 
            text_mask, 
            mel_masks,
            batch['mel_lens'],
            batch.get('speakers', None))

               
    def get_masks(self, text_lens, text_max_len, mel_lens, mel_max_len):
        text_masks = get_mask_from_lengths(text_lens, text_max_len)
        mel_masks = get_mask_from_lengths(mel_lens, mel_max_len)
        return text_masks, mel_masks
    
