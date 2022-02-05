import re
import os
from pyrsistent import v
import yaml
import json
import torch
from g2p_en import G2p
import librosa
from string import punctuation
import numpy as np

from fastspeech2.model.fastspeech2 import FastSpeech2
from .model.ptl_module import FS2TrainingModule
from .data.preprocessing.text import text_to_sequence
from .data.preprocessing.text.cmudict import CMUDict
from .data.preprocessing.audio.audio_tools import melspectrogram
from hifi_gan.inference import vocoder_inference


def open_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
    
def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(config['lexicon_path'])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, config["text"]["text_cleaners"]
        )
    )
    return torch.tensor(sequence)


def inference(text, 
              checkpoint_path=None,
              model_path=None,
              vocoder='hifi',
              speaker_wav=None, 
              control={'pitch':1.0, 'duration':1.0, 'energy':1.0},
            ):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_config = 'fastspeech2/config/data.yaml'
    model_config = 'fastspeech2/config/model.yaml'
    
    # Config
    with open(data_config) as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    audio_metadata = f'fastspeech2/data/database/{data_config["dataset"]}/processed_dataset/all_metadata.json'    
    
    with open(audio_metadata, 'r') as f:
        audio_metadata = json.loads(f.read())
    # Preprocess
    input = preprocess_english(text, data_config)
    
    # Speaker embedding
    if speaker_wav is not None: 
        speaker_wav, sr = librosa.load(speaker_wav,data_config['audio']['sample_rate'])
        speaker_mel = melspectrogram(speaker_wav,
                                    sample_rate=data_config['audio']['sample_rate'],
                                    n_fft=data_config['audio']['n_fft'],
                                    n_mels=data_config['audio']['n_mels'],
                                    f_min=data_config['audio']['f_min'],
                                    hop_length=data_config['audio']['hop_length'],
                                    win_length=data_config['audio']['win_length'],
                                    min_level_db=data_config['audio']['min_level_db'])
        
        speaker_mel = torch.tensor(speaker_mel, device=device).unsqueeze(0).transpose(1,2)
    else: speaker_emb = None
    
    # Load model
    if checkpoint_path is not None:
        fs2 = FS2TrainingModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            audio_metadata=audio_metadata).fs2
    elif model_path is not None:
        fs2 = FastSpeech2.load_state_dict(
            torch.load(model_path), 
            open_config(model_config), 
            audio_metadata)

    # Inference
    with torch.no_grad():
        fs2.eval()
        fs2.to(device)
        output = fs2.forward(phonemes=input.unsqueeze(0).to(device), 
                             speaker_embedding=speaker_emb,
                             pitch_alpha=control['pitch'],
                             energy_alpha=control['energy'],
                             duration_alpha=control['duration'])
        
    mel_postnet, mel_preds, pitch_preds, energy_preds, log_duration_preds = output 
    mel_postnet = mel_postnet.transpose(1,2)
    
    wav_paths = vocoder_inference(mel_postnet, data_config['audio']['sample_rate'], vocoder)
    
    return output, wav_paths

# TODO finish inference
# Make notebook
# Post to git
# Make blogpost