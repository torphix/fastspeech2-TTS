import os
import sys
import yaml
import shutil
import logging
import argparse
from fastspeech2.train import train
from fastspeech2.export import onnx_export
from fastspeech2.inference import inference
from fastspeech2.data.download import download_dataset
from fastspeech2.data.preprocessing.align_dataset import align_corpus
from fastspeech2.data.preprocessing.create_dataset import CreateDataset
from fastspeech2.data.preprocessing.prepare_dataset import prepare_for_alignment


def update_config_with_arg(config_path, args):
    with open(config_path, 'r') as config_f:
        config = yaml.load(config_f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        config[k] = v
    with open(config_path, 'w') as config_f:
        yaml.dump(config, config_f)

if __name__ == '__main__':
    command = sys.argv[1]    
    parser = argparse.ArgumentParser()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if command == 'fs2_make_dataset':
        data_config = 'fastspeech2/config/data.yaml'
        parser.add_argument('--dataset', required=True)
        args, leftover_args = parser.parse_known_args()
        update_config_with_arg(data_config, args)
        # download_dataset(args.dataset)
        # prepare_for_alignment(data_config)
        # align_corpus(data_config)
        logger.info('Alignment Complete, processing dataset...')
        with open(f'fastspeech2/config/data.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        create_dataset = CreateDataset(f'fastspeech2/config/data.yaml')
        create_dataset.process_aligned_corpus()
    
    elif command == 'download_dataset':
        parser.add_argument('--dataset', required=True)
        args, leftover_args = parser.parse_known_args()
        download_dataset(args.dataset)
        
    # Fastspeech 2 Methods
    elif command == 'fs2_prepare_dataset':
        data_config = 'fastspeech2/config/data.yaml'
        parser.add_argument('--dataset')
        args, leftover_args = parser.parse_known_args()
        if args.dataset is not None:
            update_config_with_arg(data_config, args)
        prepare_for_alignment(data_config)
            
    elif command == 'fs2_align_dataset':
        data_config = 'fastspeech2/config/data.yaml'
        parser.add_argument('--dataset')
        args, leftover_args = parser.parse_known_args()
        if args.dataset is not None:
            update_config_with_arg(data_config, args)
        align_corpus(data_config)
            
    elif command == 'fs2_create_dataset':
        parser.add_argument('--config_dir', default='fastspeech2/config')
        args, leftover_args = parser.parse_known_args()
        with open(f'fastspeech2/config/data.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        create_dataset = CreateDataset(f'fastspeech2/config/data.yaml')
        create_dataset.process_aligned_corpus()
            
    elif command == 'fs2_train':
        parser.add_argument('--config_dir', default='fastspeech2/config')
        parser.add_argument('--ckpt_path', required=False)
        args, leftover_args = parser.parse_known_args()
        # Load configs
        model_config = f'fastspeech2/config/model.yaml'
        data_config = f'fastspeech2/config/data.yaml'
        trainer_config = f'fastspeech2/config/trainer.yaml'
        train(model_config, 
                trainer_config, 
                data_config,
                args.ckpt_path)
        
    elif command == 'inference':
        parser.add_argument('--text', required=True,
                            help='Text to transcribe')
        parser.add_argument('--model_path',required=True, 
                            help='Look under fastspeech2/trained_models for models that are trained')
        parser.add_argument('--pitch_control', default=1.0)
        parser.add_argument('--energy_control', default=1.0)
        parser.add_argument('--duration_control', default=1.0)
        args, leftover_args = parser.parse_known_args()
               
        control = {
            'pitch':float(args.pitch_control),
            'energy':float(args.energy_control),
            'duration':float(args.duration_control),
        }
        
        inference(text=args.text, 
                  model_path=args.model_path,
                  control=control)
            
    elif command == 'flush_data':
        parser.add_argument('--text', required=True)
        args, leftover_args = parser.parse_known_args()
        # Clears out all the data from files EXCEPT: processed_data
        shutil.rmtree(os.path.abspath(f'fastspeech2/data/database/{args.name}/raw_data'))
        shutil.rmtree(os.path.abspath(f'fastspeech2/data/database/{args.name}/preprocessed_dataset'))
        os.mkdir(os.path.abspath(f'fastspeech2/data/database/{args.name}/raw_data'))
        os.mkdir(os.path.abspath(f'fastspeech2/data/database/{args.name}/preprocessed_dataset'))
            
    elif command == 'flush_wavs':
        # Clears out the output wav folder
        [os.remove(os.path.abspath(f'output_wavs/{f}')) 
         for f in os.listdir('output_wavs')]
        
    elif command == 'onnx_export':
        parser.add_argument('--model_path', required=True)
        args, leftover_args = parser.parse_known_args()
        onnx_export(args.model_path)
        
        
    else:
        print(f'''
              Command "{command}" not recognized, options are:
                - fs2_make_dataset
                - fs2_prepare_dataset
                - fs2_align_dataset
                - fs2_create_dataset
                - fs2_train
                - inference
                - flush_data
                - flush_wavs
              ''')
                  