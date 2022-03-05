import os
import torch
import json
import yaml
from fastspeech2.model.fastspeech2 import FastSpeech2


def onnx_export(state_dict_path):  
    print('Exporting Onnx model')
    # Configs
    with open('fastspeech2/config/model.yaml') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
        model_config['onnx_export'] = True
    with open('fastspeech2/config/data.yaml') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)  
    
    # Get metadata for the pitch & energy embeddings
    ds_path = f'{data_config["dataset_dir"]}/{data_config["dataset"]}/processed_dataset/all_metadata.json'
    with open(os.path.abspath(ds_path), 'r') as f:
        audio_metadata = json.loads(f.read())
        
    model = FastSpeech2(model_config['model'], audio_metadata)
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    phoneme_in = torch.ones((1, 20)).int().to(device)
    speed_control = torch.tensor(1.).to(device)
    pitch_control = torch.tensor(1.).to(device)
    energy_control = torch.tensor(1.).to(device)
    dummy_input = (phoneme_in, speed_control, pitch_control, energy_control)
    dynamic_axes = {'phonemes':{0:'batch', 1:'length'},
                    'mel_postnet': {0: 'batch', 1:'length'},
                    'mel_preds': {0:'batch', 1:'length'},
                    'pitch_preds': {0:'batch', 1:'length'},
                    'energy_preds': {0:'batch', 1: 'length'},
                    'log_duration_preds': {0:'batch', 1:'length'}
                    }
    
    input_names = ["phonemes", "speed_control", "pitch_control", "energy_control"]
    output_names = ["mel_postnet", "mel_preds", "pitch_preds", "energy_preds", "log_duration_preds"]
    
    torch.onnx.export(model,
                      dummy_input,
                      "fastspeech2/saved_models/fs2.onnx",
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      export_params=True,
                    #   operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                      opset_version=13)
    
    
