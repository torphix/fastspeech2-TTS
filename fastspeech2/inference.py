import re
import os
import yaml
import json
import torch
import logging
import numpy as np
from g2p_en import G2p
from scipy.io import wavfile
from natsort import natsorted
from string import punctuation
from vocoder.api import get_vocoder, vocoder_infer
from fastspeech2.model.fastspeech2 import FastSpeech2
from .data.preprocessing.text import text_to_sequence

logger = logging.getLogger()

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
              model_path=None,
              control={'pitch':1.0, 'duration':1.0, 'energy':1.0}):
    
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
    
    # Load model
    if model_path is not None:
        fs2 = FastSpeech2(
                open_config(model_config)['model'], 
                audio_metadata)
    else:
        model_path = natsorted([f'fastspeech2/trained_models/{file}'
                                for file in os.listdir('fastspeech2/trained_models')])[-1]
        logger.info(f'Model path not provided, loading {model_path}')
        
        with open(model_config, 'r') as f:
            model_config = yaml.loader(f.read(), Loader=yaml.FullLoader)
        fs2 = FastSpeech2.load_state_dict(
            torch.load(''), 
            open_config(model_config), 
            audio_metadata)
    # Inference
    with torch.no_grad():
        fs2.eval()
        fs2.to(device)
        output = fs2.forward(phonemes=input.unsqueeze(0).to(device), 
                             pitch_alpha=control['pitch'],
                             energy_alpha=control['energy'],
                             duration_alpha=control['duration'])
        
    mel_postnet, mel_preds, pitch_preds, energy_preds, log_duration_preds = output 
    mel_postnet = mel_postnet.transpose(1,2)
    vocoder = get_vocoder(device)
    wav = vocoder_infer(mel_postnet, vocoder)[0]
    wavfile.write('output.wav', 22050, wav.squeeze(0))
    logger.info('Wrote wav file to output.wav')
