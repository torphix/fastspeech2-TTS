'''Creates output JSON file where each entry is a datapoint'''
import os
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
        self.normalization_type = config['audio']['norm_type']
        assert self.normalization_type in ['min_max', 'mean_std'], \
             'Normalization min_max or mean_std (scalar) supported'
        # Averaging
        self.pitch_feature_level = config['audio']['pitch_feature_level']
        self.energy_feature_level = config['audio']['energy_feature_level']
        assert self.pitch_feature_level in ['mel_frame', 'phoneme_frame'], \
            'Feature level for pitch must be mel_frame or phoneme_frame'
        assert self.energy_feature_level in ['mel_frame', 'phoneme_frame'], \
            'Feature level for energy must be mel_frame or phoneme_frame'
            
        # Normalizations
        self.pitch_normalization = config["audio"]["normalization"]
        self.energy_normalization = config["audio"]["normalization"]
        # STFT && Mel
        self.STFT = TacotronSTFT(
            self.n_fft, self.hop_length, 
            self.win_length, self.n_mels, 
            self.sampling_rate, self.f_min, self.f_max)
        
        if self.config['speaker_embedding']['embed']:
            print('Speaker Embedding set to true, will take longer to process dataset..')
            # Speaker Embedding
            from speaker_encoder.compute_embedding import SpeechEmbedding
            from .utils import load_speaker_encoder_config
            # Load model
            speaker_embedding_config = load_speaker_encoder_config(
                self.config['speaker_embedding']['config_path'])
            
            self.speech_emb = SpeechEmbedding(
                speaker_embedding_config, 
                self.config['speaker_embedding']['model_path'])
        

    def process_aligned_corpus(self):
        self.pitch_scaler = StandardScaler()
        self.energy_scaler = StandardScaler()
        self.mel_scaler = StandardScaler()
        
        for speaker in tqdm(os.listdir(self.input_data_dir)):
            
            wav_txt_dir = f"{self.input_data_dir}/{speaker}"
            tg_speaker_dir = f"{self.input_data_dir}/{speaker}/TextGrid"
            
            output_data_dir = f'{self.output_data_dir}/{speaker}'
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
                
                data_points.append({
                    'id':basename,
                    'phonemes':phonemes,
                    'raw_text':raw_text,
                    'mel':mel_fname, 
                    'duration':duration_fname,
                    'energy':energy_fname,
                    'pitch':pitch_fname,
                })
                if idx == 24: break
            with open(f'{output_data_dir}/{speaker}_data.json', 'w') as json_datafile:
                json_datafile.write(json.dumps(
                    {
                        'root':output_data_dir,
                        'data':data_points
                    }))
                
            self.compute_metadata(output_data_dir)
            print("Dataset creation complete!")
        self.compute_all_metadata(self.output_data_dir)
        print('Metadata for entire dataset calculated')
        print('Dataset creation completed.')
        
    def compute_all_metadata(self, root_dir):
        '''
        Computes metadata across all folders, saves to yaml 
        then normalizes the data across all files
        '''
        pitch_mean, pitch_std = self.pitch_scaler.mean_[0], self.pitch_scaler.scale_[0]
        energy_mean, energy_std = self.energy_scaler.mean_[0], self.energy_scaler.scale_[0]
        mel_mean, mel_std = self.mel_scaler.mean_[0], self.mel_scaler.scale_[0]

        # Normalize
        print('Normalizing data...')
        for speaker in tqdm(os.listdir(f'{root_dir}')):
            if speaker == 'all_metadata.json': continue
            print('Normalizing Energy')
            energy_min, energy_max = self.normalize(
                f'{root_dir}/{speaker}/energy',
                energy_mean,
                energy_std)
            print('Normalizing Pitch')
            pitch_min, pitch_max = self.normalize(
                    f'{root_dir}/{speaker}/pitch',
                    pitch_mean,
                    pitch_std)
        with open(f'{root_dir}/all_metadata.json', 'w') as f:
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
                'mel':{
                        # 'max': mel_max.astype(float),
                        # 'min': mel_min.astype(float),
                        'mean':mel_mean.astype(float),
                        'std':mel_std.astype(float),
                    },
                }
            ))
        print('Data normalization complete!')
                
    def compute_metadata(self, in_dir):
        '''Computes metadata for each speaker and updates global values'''
        print('Creating Metadata...')
        # Normalizes saved pitch & energy numpy files
        print('Normalizing pitch & energy files...')
        # Standard scaler norm
        self.update_data_values(f"{in_dir}/pitch", 'pitch')
        self.update_data_values(f"{in_dir}/energy", 'energy')
        self.update_data_values(f"{in_dir}/mel", 'mel')

    def update_data_values(self, in_dir, name):
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = np.load(filename)
            if name == 'pitch':
                self.pitch_scaler.partial_fit(values.reshape(-1, 1))
            elif name == 'energy':
                self.energy_scaler.partial_fit(values.reshape(-1, 1))
            elif name == 'mel':
                self.mel_scaler.partial_fit(values.reshape(-1, 1))
        
    def normalize(self, in_file, mean, std):
        max_v = np.finfo(np.float64).min
        min_v = np.finfo(np.float64).max
        for file in tqdm(os.listdir(in_file)):
            values = np.load(f'{in_file}/{file}')
            if self.normalization_type == 'mean_std':
                values = (values - mean) / std
            elif self.normalization_type == 'min_max':
                values = (values - min_v) / (max_v - min_v)
            np.save(f'{in_file}/{file}', values)
            max_v = max(max_v, max(values.reshape(-1,1)))
            min_v = min(min_v, min(values.reshape(-1,1)))
        return min_v.item(), max_v.item()
    
    # def normalize(self, in_dir, mean, std):
    #     max_value = np.finfo(np.float64).min
    #     min_value = np.finfo(np.float64).max
    #     for filename in os.listdir(in_dir):
    #         filename = os.path.join(in_dir, filename)
    #         values = (np.load(filename) - mean) / std
    #         np.save(filename, values)

    #         max_value = max(max_value, max(values))
    #         min_value = min(min_value, min(values))

    #     return min_value.item(), max_value.item()
    
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
    
    
def embed_speaker_for_processed_dataset(processed_dataset_path, config):
    '''
    For extracting speakers embedding when dataset (mel, pitch, energy)
    has already been computed, saves you from reprocessing everything
    '''
    # Speaker Embedding
    from speaker_encoder.compute_embedding import SpeechEmbedding
    from .utils import load_speaker_encoder_config


    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Load model
    speaker_embedding_config = load_speaker_encoder_config(
        config['speaker_embedding']['config_path'])
    speech_emb = SpeechEmbedding(
        speaker_embedding_config, 
        config['speaker_embedding']['model_path'])
    # Get speakers
    print('Loading speakers')
    for speaker in tqdm(os.listdir(processed_dataset_path)):
        root = f'{processed_dataset_path}/{speaker}'
        os.makedirs(f'{root}/speaker', exist_ok=True) 
        # Load data
        with open(f'{root}/{speaker}_data.json', 'r') as f:
            data_f = json.loads(f.read())
        # Save embedding
        print('Loading data...')
        for data_point in tqdm(data_f['data']):
            mel = np.load(f'{root}/{data_point["mel"]}')
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            emb = speech_emb.model.compute_embedding(
                torch.tensor(mel).unsqueeze(0).transpose(1,2).to(device))
            
            f_name = f'{root}/speaker/{data_point["id"]}-speaker.npy' 
            np.save(f_name, emb.cpu())
            data_point['speaker'] = f'speaker/{data_point["id"]}-speaker.npy'
            print(root, data_point['id'])
        with open(f'{root}/{speaker}_data.json', 'w') as f:
            data_f = f.write(json.dumps(data_f))
            
        print(f'Completed speaker {speaker}..')
        

    print('Speaker embedding computation complete!')
            
        

