import os
import sys
import yaml
import shutil
import argparse
from fastspeech2.data.preprocessing.align_dataset import align_corpus
from fastspeech2.data.preprocessing.prepare_dataset import prepare_for_alignment
from fastspeech2.data.preprocessing.create_dataset import CreateDataset
from fastspeech2.train import train
from fastspeech2.inference import inference
from fastspeech2.data.preprocessing.create_dataset import embed_speaker_for_processed_dataset
from fastspeech2.data.download import download_dataset
from hifi_gan.inference import vocoder_inference


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
    if command == 'fs2_make_dataset':
        try:
            data_config = 'fastspeech2/config/data.yaml'
            parser.add_argument('--dataset', required=True)
            args, leftover_args = parser.parse_known_args()
            update_config_with_arg(data_config, args)
            download_dataset(args.dataset)
            prepare_for_alignment(data_config)
            align_corpus(data_config)
            with open(f'{args.config_dir}/data.yaml', 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            create_dataset = CreateDataset(f'{args.config_dir}/data.yaml')
            create_dataset.process_aligned_corpus()
        except:
            print('Error running joint commands, try running commands separately as specified in README.md and open an issue if error persists.')
    
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
        with open(f'{args.config_dir}/data.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        create_dataset = CreateDataset(f'{args.config_dir}/data.yaml')
        create_dataset.process_aligned_corpus()
            
    elif command == 'fs2_train':
        parser.add_argument('--config_dir', default='fastspeech2/config')
        args, leftover_args = parser.parse_known_args()
        # Load configs
        model_config = f'{args.config_dir}/model.yaml'
        data_config = f'{args.config_dir}/data.yaml'
        trainer_config = f'{args.config_dir}/trainer.yaml'
        train(model_config, 
                trainer_config, 
                data_config)
        
    elif command == 'inference':
        parser.add_argument('--text', required=True)
        parser.add_argument('--checkpoint')
        parser.add_argument('--model_path')
        parser.add_argument('--vocoder', default='hifi')
        parser.add_argument('--pitch_control', default=1.0)
        parser.add_argument('--energy_control', default=1.0)
        parser.add_argument('--duration_control', default=1.0)
        args, leftover_args = parser.parse_known_args()
        
        if args.checkpoint is None and args.model_path is None:
            print('Checkpoint and model path not provided, attempting to load checkpoint from fastspeech2/checkpoints ...')
            if os.path.isfile('fastspeech2/checkpoints/epoch=119-step=96695.ckpt'):
                print('Found checkpoint, loading into memory...')
                args.checkpoint = 'fastspeech2/checkpoints/epoch=119-step=96695.ckpt'
            else:
                raise ValueError('Unable to autoload checkpoint, please provide --checkpoint path or --model_path')
        
        control = {
            'pitch':float(args.pitch_control),
            'energy':float(args.energy_control),
            'duration':float(args.duration_control),
        }
        
        inference(text=args.text, 
                  checkpoint_path=args.checkpoint,
                  model_path=args.model_path,
                  vocoder=args.vocoder,
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
        
    else:
        print(f'''
              Command "{command}" not recognized, options are:
                - fs2_prepare_dataset
                - fs2_align_dataset
                - fs2_create_dataset
                - fs2_train
                - inference
                - flush_data
                - flush_wavs
              ''')
                  