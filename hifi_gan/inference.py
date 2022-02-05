import os
import torch
import json
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from .model import Generator
import time

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def vocoder_inference(
    mel,
    sample_rate=22050,
    vocoder_type='hifi'):
    
    print('Converting Melspectrogram to wav file')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if vocoder_type == 'melgan':
        vocoder = torch.hub.load("descriptinc/melgan-neurips", "load_melgan", "linda_johnson")
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
        mel = torch.tensor(mel / np.log(10))
        with torch.no_grad():
            wavs = vocoder.inverse(mel)
            
    elif vocoder_type == 'hifi':
        with open('hifi_gan/config.json', 'r') as f:
            config = AttrDict(json.loads(f.read()))
        vocoder = Generator(config)
        vocoder.load_state_dict(
            torch.load('hifi_gan/saved_models/hifi_generator_LJSpeech.pth.tar')['generator'])
        vocoder.eval()
        vocoder.to(device)
        with torch.no_grad():
            wavs = vocoder.forward(
                torch.tensor(mel, device=device))
    else:
        raise ValueError('--vocoder_type argument must == melgan or hifi')
            
    # Un-normalize
    wavs = (wavs.cpu().numpy() * 32768.0).astype("int16")
    wavs = [wav for wav in wavs]
    outputs = []
    output_f = f'output_wavs' 
    os.makedirs(output_f, exist_ok=True)
    # Save
    for wav in tqdm(wavs):
        output_path = f'{output_f}/{int(time.time())}.wav'
        wavfile.write(output_path, sample_rate, wav.squeeze(0))
        outputs.append(output_path)
    print(f'TTS generation complete! outputs saved to {output_f}')
    return outputs