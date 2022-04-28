'''Creates output JSON file where each entry is a datapoint'''
import os
import shutil
import tgt
import yaml
import torch
import json
import librosa
import numpy as np
from tqdm import tqdm
import pyworld as pw
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .audio import audio_tools

from .audio.stft import TacotronSTFT


class CreateDataset():
    def __init__(self, config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        self.config = config
        # I/O directory paths
        self.config_path = config_path
        self.dataset_type = config['dataset']
        self.root = os.path.abspath(config['dataset_dir'])
        self.input_data_dir = f"{self.root}/{self.dataset_type}/preprocessed_dataset"
        self.output_data_dir = f"{self.root}/{self.dataset_type}/processed_dataset"
        # Audio Hyper parameters
        self.hop_length = config["audio"]["hop_length"]
        self.sampling_rate = config["audio"]["sample_rate"]
        self.n_fft = config['audio']['n_fft']
        self.n_mels = config['audio']['n_mels']
        self.f_min = config['audio']['f_min']
        self.f_max = config['audio']['f_max']
        self.hop_length = config['audio']['hop_length']
        self.win_length = config['audio']['win_length']
        self.min_level_db = config['audio']['min_level_db']
        self.max_wav_value = config["audio"]["max_wav_value"]
        self.normalisation_type = config['audio']['norm_type']
        assert self.normalisation_type in ['min_max', 'mean_std'], \
             'normalisation min_max or mean_std (scalar) supported'
        # Averaging
        self.pitch_feature_level = config['audio']['pitch_feature_level']
        self.energy_feature_level = config['audio']['energy_feature_level']
        assert self.pitch_feature_level in ['mel_frame', 'phoneme_frame'], \
            'Feature level for pitch must be mel_frame or phoneme_frame'
        assert self.energy_feature_level in ['mel_frame', 'phoneme_frame'], \
            'Feature level for energy must be mel_frame or phoneme_frame'
            
        # normalisations
        self.pitch_normalisation = config["audio"]["normalisation"]
        self.energy_normalisation = config["audio"]["normalisation"]
        # STFT && Mel
        self.STFT = TacotronSTFT(
            self.n_fft, self.hop_length, 
            self.win_length, self.n_mels, 
            self.sampling_rate, self.f_min, self.f_max)

        

    def process_aligned_corpus(self):
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        
        for speaker in tqdm(os.listdir(self.input_data_dir)):
            
            wav_txt_dir = f"{self.input_data_dir}/{speaker}"
            tg_speaker_dir = f"{self.input_data_dir}/{speaker}/TextGrid"
            
            output_data_dir = f'{self.output_data_dir}/{speaker}'
            shutil.rmtree(output_data_dir, ignore_errors=True)
            os.makedirs(f"{output_data_dir}/duration", exist_ok=True)
            os.makedirs(f"{output_data_dir}/pitch", exist_ok=True)
            os.makedirs(f"{output_data_dir}/energy", exist_ok=True)
            os.makedirs(f"{output_data_dir}/mel", exist_ok=True)
            os.makedirs(f"{output_data_dir}/speaker_emb", exist_ok=True)
            
            data_points = []
            for idx, tg_name in enumerate(tqdm(os.listdir(f'{tg_speaker_dir}'))):
                basename = tg_name.split(".")[0]
                values = \
                    self.process_datapoint(wav_txt_dir, tg_speaker_dir, basename)
                if values is None: continue
                
                if self.config['speaker_embedding']['preprocess']:
                    phonemes, raw_text, mel, pitch, energy, duration, speaker_embedding = values
                    speaker_emb_fname = f"speaker_emb/{basename}-speaker-emb.npy"       
                    np.save(f'{output_data_dir}/{speaker_emb_fname}', speaker_embedding)
                else:
                    phonemes, raw_text, mel, pitch, energy, duration, _ = values
                # Save files
                duration_fname = f"duration/{basename}-duration.npy"
                np.save(f'{output_data_dir}/{duration_fname}', duration)                    
                pitch_fname = f"pitch/{basename}-pitch.npy"
                np.save(f'{output_data_dir}/{pitch_fname}', pitch)                    
                energy_fname = f"energy/{basename}-energy.npy"
                np.save(f'{output_data_dir}/{energy_fname}', energy)
                mel_fname = f"mel/{basename}-mel.npy"
                np.save(f'{output_data_dir}/{mel_fname}', mel)
                
                
                if len(pitch) > 0: 
                    pitch = self.remove_outlier(pitch)
                    pitch_scaler.partial_fit(pitch.reshape((-1,1)))
                if len(energy) > 0:
                    energy = self.remove_outlier(energy)
                    energy_scaler.partial_fit(energy.reshape((-1,1)))
                
                
                data_points.append({
                    'id':basename,
                    'phonemes':phonemes,
                    'raw_text':raw_text,
                    'mel':mel_fname, 
                    'duration':duration_fname,
                    'energy':energy_fname,
                    'pitch':pitch_fname,
                })
                # if idx == 24: break
            with open(f'{output_data_dir}/{speaker}_data.json', 'w') as json_datafile:
                json_datafile.write(json.dumps(
                    {
                        'root':output_data_dir,
                        'data':data_points
                    }))
        
        print('Normalising data')
        pitch_mean, pitch_std = pitch_scaler.mean_[0], pitch_scaler.scale_[0]
        energy_mean, energy_std = energy_scaler.mean_[0], energy_scaler.scale_[0]
        pitch_min, pitch_max = self.normalize(f'{output_data_dir}/pitch/', pitch_mean, pitch_std)
        energy_min, energy_max = self.normalize(f'{output_data_dir}/energy/', energy_mean, energy_std)
        print('Saving meta-data')
        with open(f'{output_data_dir}/all_metadata.json', 'w') as f:
           f.write(json.dumps(
                {
                'pitch':{
                        'max': pitch_max,
                        'min': pitch_min,
                        'mean':pitch_mean.astype(float),
                        'std':pitch_std.astype(float),
                    },
                'energy':{
                        'max': energy_max,
                        'min': energy_min,
                        'mean':energy_mean.astype(float),
                        'std':energy_std.astype(float),
                    },
                }
            ))
        # Would average over all speakers here if doing multi tts
        shutil.copyfile(f'{output_data_dir}/all_metadata.json',
                        f'{self.output_data_dir}/all_metadata.json')
        print('Data normalisation complete!')
                
    def normalize(self, in_file, mean, std):
        max_v = np.finfo(np.float64).min
        min_v = np.finfo(np.float64).max
        for file in tqdm(os.listdir(in_file)):
            values = np.load(f'{in_file}/{file}')
            values = (values - mean) / std
            np.save(f'{in_file}/{file}', values)
            max_v = max(max_v, max(values))
            min_v = min(min_v, min(values))
        return min_v.item(), max_v.item()
    
    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)
        return values[normal_indices]
    
    def process_datapoint(self, wav_txt_dir, tg_speaker_dir, basename):
        wav_path = f"{wav_txt_dir}/{basename}.wav"
        text_path = f"{wav_txt_dir}/{basename}.lab"
        tg_path = f"{tg_speaker_dir}/{basename}.TextGrid"
        # Get Alignments
        tg = tgt.io.read_textgrid(tg_path)
        phonemes, duration, start, end = self.get_alignment(tg.get_tier_by_name("phones"))
        
        if start >= end: return None
        
        # Get Text
        with open(text_path, 'r') as text_f:
            raw_txt = text_f.readline().strip("\n")
        # Trim wav files, compute audio features
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate*start):int(self.sampling_rate*end)
        ].astype(np.float32)
        # Pitch
        # Compute fundamental frequency
        pitch = self.get_pitch(wav)
        pitch = pitch[:sum(duration)]

        if np.sum(pitch != 0) <= 1:
            return None
        # Mel
        mel, energy = self.get_mel(wav)
        mel = mel[:, : sum(duration)]
        energy = energy[:sum(duration)]
        
        speaker_embedding = None
        if self.config['speaker_embedding']['preprocess']:
            speaker_embedding = self.get_speaker_encoding(mel)
        
        if self.energy_feature_level == 'phoneme_frame':
            energy = audio_tools.energy_phoneme_averaging(energy, duration)
        if self.pitch_feature_level == 'phoneme_frame':
            pitch = audio_tools.pitch_phoneme_averaging(pitch, duration)
        return phonemes, raw_txt, mel, pitch, energy, duration, speaker_embedding

    
    def get_mel(self, wav):
        audio = torch.clip(torch.FloatTensor(wav).unsqueeze(0), -1, 1)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec, energy = self.STFT.mel_spectrogram(audio)
        melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
        energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
        return melspec, energy
    
    def get_pitch(self, wav):
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        return pitch  
    
    def get_speaker_encoding(self, mel):
        embedding = self.speech_emb.compute_embedding(mel)
        return embedding

    def get_alignment(self, tier):
        silent_phonemes = ["sil", "sp", "spn"]
        phonemes = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            # Trim leading silences
            if phonemes == []:
                if p in silent_phonemes:
                    continue
                else:
                    start_time = s
            if p not in silent_phonemes:
                # For ordinary phonemes
                phonemes.append(p)
                end_time = e
                end_idx = len(phonemes)
            else:
                # For silent phonemes
                phonemes.append(p)
            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)))
        # Trim tailing silences
        phonemes = phonemes[:end_idx]
        durations = durations[:end_idx]
        return phonemes, durations, start_time, end_time
    
