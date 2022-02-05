import os
import yaml
import torch
import json
from datetime import datetime
import pytorch_lightning as ptl
from .data.data import get_dataloaders
from .model.ptl_module import FS2TrainingModule
from pytorch_lightning.loggers import TensorBoardLogger



def train(model_config, trainer_config, data_config):
    # Load configs
    with open(model_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(trainer_config) as f:
        trainer_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(data_config) as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    # Get data
    train_paths = []        
    rootdir = f'{data_config["dataset_dir"]}/{data_config["dataset"]}/processed_dataset'
    for speaker_file in os.listdir(rootdir):
        if os.path.isdir(f'{rootdir}/{speaker_file}'):
            for file in os.listdir(os.path.abspath(f'{rootdir}/{speaker_file}')):
                if file == f'{speaker_file}_data.json':
                    train_paths.append(
                        os.path.abspath(f'{rootdir}/{speaker_file}/{file}'))
            
    # Get dataloaders
    train_dl, val_dl = get_dataloaders(
                        train_paths, 
                        data_config['text']['text_cleaners'],
                        config['dataloader'],
                        config['speaker_embedding']['type'])
    
    # Get metadata for the pitch & energy embeddings
    ds_path = f'{data_config["dataset_dir"]}/{data_config["dataset"]}/processed_dataset/all_metadata.json'
    with open(os.path.abspath(ds_path), 'r') as f:
        audio_metadata = json.loads(f.read())
        
    # Train
    fs2_module = FS2TrainingModule(model_config=model_config, 
                                   audio_metadata=audio_metadata,
                                   train_dl=train_dl, 
                                   val_dl=val_dl)
     
    ckpt_path = trainer_config['checkpoint_path']
    trainer_config.pop('checkpoint_path')
    
    logger = TensorBoardLogger("fastspeech2/checkpoints", 'fs2')
    trainer = ptl.Trainer(**trainer_config,
                          logger=logger)
    trainer.fit(
        fs2_module,
        ckpt_path=ckpt_path)
    
    torch.save(trainer.model.state_dict(), f'fastspeech2/saved_models/fs2_model.pth.{datetime.utcnow()}.tar')