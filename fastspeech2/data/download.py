import os
import wget
import tarfile
import tqdm

def download_dataset(name):
    data_root = 'fastspeech2/data/database'
    if name == 'LJSpeech':
        if os.path.isfile(f'{data_root}/LJSpeech/LJSpeech-1.1.tar.bz2'):
            print('LJSpeech corpus already downloaded.. skipping to extraction')
        else:
            print('Downloading LJSpeech corpus')
            wget.download('https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
                          out=f'{data_root}/LJSpeech')
        print('Extracting LJSpeech corpus, this usually takes a few minutes')
        tarball = tarfile.open(f'{data_root}/LJSpeech/LJSpeech-1.1.tar.bz2', 'r:bz2') 
        tarball.extractall(f'{data_root}/LJSpeech/raw_data')
        tarball.close()
        print('Extraction complete')
            
    else: 
        raise ValueError('Only dataset LJSpeech supported for download right now.')