import os
import io
import wget
import tarfile
import logging


logger = logging.getLogger()

def download_dataset(name):
    logging.info('Loading')
    data_root = 'fastspeech2/data/database'
    if name == 'LJSpeech':
        if os.path.isfile(f'{data_root}/LJSpeech/LJSpeech-1.1.tar.bz2'):
            logging.info('LJSpeech corpus already downloaded.. skipping to extraction')
        else:
            logging.info('Downloading LJSpeech corpus \n')
            wget.download('https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
                          out=f'{data_root}/LJSpeech')
        logging.info('Extracting LJSpeech corpus, this usually takes a few minutes, Please wait... \n')
        tarball = tarfile.open(f'{data_root}/LJSpeech/LJSpeech-1.1.tar.bz2', 'r:bz2') 
        os.makedirs(f'{data_root}/LJSpeech/raw_data', exist_ok=True)
        tarball.extractall(f'{data_root}/LJSpeech/raw_data')
        tarball.close()
        logging.info('Extraction complete')
            
    else: 
        raise ValueError('Only dataset LJSpeech supported for download right now.')
    
    
    