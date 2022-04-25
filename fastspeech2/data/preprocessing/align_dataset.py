from asyncio.log import logger
import os 
import yaml
import logging
import subprocess
from tqdm import tqdm
 
logger = logging.getLogger(__name__)
 
def align_corpus(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config['dataset'] == 'LibriSpeech':
        align_libri_corpus(config_path)
    elif config['dataset'] == 'LJSpeech':
        align_lj_corpus(config_path)
    else:
        raise ValueError('LibriSpeech, LJSpeech or Custom dataset supported')
 
def align_libri_corpus(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    root_dir = os.path.abspath(f"{config['dataset_dir']}/preprocessed_dataset")
    output_dir = f'{root_dir}/TextGrid'
    in_dirs = os.listdir(root_dir)
    in_dirs = [f'{root_dir}/{d}' for d in in_dirs]
    os.makedirs(output_dir, exist_ok=True)
    try:
        subprocess.run(['mfa', 'model', 'download', 'acoustic', 'english'])
        subprocess.run(['mfa', 'model', 'download', 'dictionary', 'english'])
    except:
        pass
    
    for dir in tqdm(in_dirs):
        speaker_id = dir.split("/")[-1]
        try:
            subprocess.run(['mfa', 'validate', dir, 'english', 'english'])
            subprocess.run(['mfa', 'align', dir, 'english', 'english', f'{output_dir}/{speaker_id}', '--clean'])
        except:
            print(f'Error attempting to align {dir}.. skipping to next one')


def align_lj_corpus(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    root_dir = os.path.abspath(f"{config['dataset_dir']}/LJSpeech/preprocessed_dataset/1")
    output_dir = f'{root_dir}/TextGrid'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print('Attempting to download mfa acoustic & dictionary')
        subprocess.run(['mfa', 'model', 'download', 'acoustic', 'english'])
        subprocess.run(['mfa', 'model', 'download', 'dictionary', 'english'])
    except:
        pass
    
    try:
        logger.info('Validating dataset, do not abort command...')
        subprocess.run(['mfa', 'validate', root_dir, 'english', 'english'])
        logger.info('Aligning dataset, do not abort command...')
        subprocess.run(['mfa', 'align', root_dir, 'english', 'english', f'{output_dir}'])
    except:
        print(f'Error attempting to align {root_dir}.. skipping to next one')