import os
import json
import torch
import numpy as np
from .utils import pad
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import random_split

from .preprocessing.text import text_to_sequence


class FS2Dataset(Dataset):
    def __init__(self, dataset_path, text_cleaners, speaker_emb):
        super().__init__()
        with open(dataset_path) as dataset_f:
            data = json.loads(dataset_f.read())
        
        self.data = data['data']
        self.root_dir = data['root']
        self.speaker_emb = speaker_emb
        self.text_cleaners = text_cleaners
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        datapoint =
        {
            'id':basename,
            'phonemes':phonemes,
            'raw_text':raw_text,
            'mel':mel_fname,
            'duration':duration_fname,
            'energy':energy_fname,
            'pitch':pitch_fname
            'speaker': speaker_id
        }
        '''
        datapoint = self.data[idx]
        datapoint['raw_text'] = self.data[idx]['raw_text']
        datapoint['duration'] = torch.tensor(
            np.load(f"{self.root_dir}/{datapoint['duration']}"))
        datapoint['pitch'] = torch.tensor(
            np.load(f"{self.root_dir}/{datapoint['pitch']}"))
        datapoint['mel'] = torch.tensor(
            np.load(f"{self.root_dir}/{datapoint['mel']}"))
        datapoint['energy'] = torch.tensor(
            np.load(f"{self.root_dir}/{datapoint['energy']}"))
        
        datapoint['phonemes'] = torch.tensor(
            text_to_sequence("{" + f"{' '.join(datapoint['phonemes'])}" + "}", 
                             cleaner_names=self.text_cleaners))
        
        if self.speaker_emb == 'quant':
            datapoint['speaker'] = torch.tensor(int(self.root_dir.split("/")[-1]))
        elif self.speaker_emb == 'lstm':
            datapoint['speaker'] = np.load(f"{self.root_dir}/{datapoint['speaker']}")
        return datapoint

def collate_fn(batch):
    durations=[]
    pitchs=[]
    mels=[]
    energys=[]
    phonemes=[]
    text_lens = []
    mel_lens = []
    speakers = []
    raw_texts = []
    for item in batch:
        raw_texts.append(item['raw_text'])
        durations.append(item['duration'])
        pitchs.append(item['pitch'])
        energys.append(item['energy'])
        mels.append(item['mel'])
        phonemes.append(item['phonemes'])
        text_lens.append(item['phonemes'].shape[0])
        mel_lens.append(item['mel'].shape[1])
        speakers.append(item.get('speaker', None))
    batch = {
        'durations':pad(durations),
        'pitchs':pad(pitchs),
        'mels':pad(mels),
        'energies':(pad(energys)),
        'phonemes':pad(phonemes),
        'text_lens': torch.tensor(text_lens),
        'mel_lens': torch.tensor(mel_lens),
        'text_max_len':max(text_lens),
        'mel_max_len':max(mel_lens),
        'raw_text': raw_texts,
        }
    if speakers[0] is not None:
        batch['speakers'] = torch.tensor(speakers) 
    return batch
        
def concat_dataset(dataset_paths, text_cleaners, speaker_emb):
    datasets = [FS2Dataset(ds, text_cleaners, speaker_emb) for ds in dataset_paths]
    return ConcatDataset(datasets)


def get_dataloaders(
    dataset_paths, 
    text_cleaners, 
    dataloader_config,
    speaker_emb,
    ):
    split = dataloader_config['split']
    dataloader_config.pop('split')
    
    dataset = concat_dataset(dataset_paths, text_cleaners, speaker_emb)
    # Split Dataset
    split = [int(dataset.__len__()/100 * split[0]), int(dataset.__len__()/100 * split[1])]
    if sum(split) != dataset.__len__(): 
        split[0] += dataset.__len__() - sum(split)
    # Get Dataloaders
    train_ds, val_ds = random_split(dataset, split)
    train_dl = DataLoader(train_ds, **dataloader_config,
                          collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, **dataloader_config,
                        collate_fn=collate_fn)
    return train_dl, val_dl

