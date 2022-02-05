import os
import yaml
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from .text import _clean_text

'''Methods for creating required MFA format'''
def prepare_for_alignment(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    if config['dataset'] == 'LibriSpeech':
        prepare_for_alignment_librispeech(config_path)
    elif config['dataset'] == 'LJSpeech':
        prepare_for_alignment_ljspeech(config_path)
    else: raise ValueError('dataset types LibriSpeech or LJSpeech required in fastspeech2/config/data.yaml')


def prepare_for_alignment_librispeech(config_path):
    
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    '''
    MFA requires corpus to be in format
    .wav, .lab, 1-4-1, audio-transcript
    '''
    def get_txt_files_recursively(root):
        for root, dirs, files in os.walk(root):
            for file in files:
                if file.endswith('.txt'):
                    yield file
    
    # Normalizes wavs, cleans && standardizes text, arranged output_dir->speaker_id
    in_dirs = config["data"]["corpus_path"]
    out_dir = os.path.abspath(config["data"]["preprocessed_dataset_dir"])
    sampling_rate = config["preprocessing"]["audio"]["sample_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    print('Preparing Dataset for Alignment...')
    for idx, in_dir in enumerate(in_dirs):
        for speaker in tqdm(os.listdir(in_dir)):
            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            print(os.path.join(out_dir, speaker))
            with open (os.path.join(out_dir, speaker, "cleaned_transcripts.txt"), "w") as transcript_file:
        
                for txt_file in get_txt_files_recursively(os.path.join(in_dir, speaker)):
                    chapter = txt_file.split("-")[1].split(".")[0]
                    with open(f'{in_dir}/{speaker}/{chapter}/{txt_file}') as source_t:
                        lines = source_t.readlines()
                    wav_ids = [line.split(" ")[0] for line in lines]
                    texts = [" ".join(line.split(" ")[1:]).strip("\n") for line in lines]
                    clean_texts = [_clean_text(text, cleaners) for text in texts]                    
                        
                    # Generate Cleaned Dataset
                    for idx, wav_id in enumerate(tqdm(wav_ids)):
                        transcript_file.write(f"{wav_id} {clean_texts[idx]}\n")
                        # Normalize Wav File
                        chapter = wav_id.split(" ")[0].split("-")[1]
                        wav_path = os.path.join(in_dir, speaker, chapter, f"{wav_id}.flac")
                        wav, _ = librosa.load(wav_path, sampling_rate)
                        wav = wav / max(abs(wav)) * max_wav_value
                        wavfile.write(
                            os.path.join(out_dir, speaker, f"{wav_id}.wav"),
                            sampling_rate,
                            wav.astype(np.int16))
                        with open(
                            os.path.join(out_dir, speaker, f"{wav_id}.lab"), 'w') as lab_f:
                                lab_f.write(clean_texts[idx])
                            
        print(f'Processed dataset {idx}.. ')
    print('Dataset processing complete')


def prepare_for_alignment_ljspeech(config_path):
    '''
    MFA requires corpus to be in format
    .wav, .lab, 1-4-1, audio-transcript
    '''    
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    sampling_rate = config["audio"]["sample_rate"]
    max_wav_value = config["audio"]["max_wav_value"]
    
    root_dir = os.path.abspath(f"{config['dataset_dir']}/LJSpeech")
    raw_data_dir = os.path.abspath(f"{config['dataset_dir']}/LJSpeech/raw_data/LJSpeech-1.1")
    metadata_path = f"{raw_data_dir}/metadata.csv"
    audio_path = f"{raw_data_dir}/wavs"
    output_path = f"{root_dir}/preprocessed_dataset/1"
    
    os.makedirs(output_path, exist_ok=True) 
    with open(metadata_path) as f:
        data = f.readlines()
        
    for line in tqdm(data):
        id = line.split("|")[0]
        text = line.split("|")[-1]

        if os.path.isfile(f"{output_path}/{id}.lab") == False:
            with open(f"{output_path}/{id}.lab", 'w') as lab_f:
                lab_f.write(text)
                
        src_wav_path = f"{audio_path}/{id}.wav"
        tgt_wav_path = f"{output_path}/{id}.wav"
        if os.path.isfile(tgt_wav_path) == False:
            wav, _ = librosa.load(src_wav_path, sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(tgt_wav_path, sampling_rate, wav.astype(np.int16))
        


def prepare_for_alignment_custom(config_path):
    # TODO this method should process your raw data into useable values
    '''
    Your raw dataset --> preprocessed_dataset
                            - speaker_id (put 1 if single speaker dataset)
                                - .lab file .wav file one for one per datapoint
    Note ensure to put utterances in seperate dirs per speaker event if single speaker dataset
    '''
    pass