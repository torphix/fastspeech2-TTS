import os
import re
import json
import yaml
import torch
import numpy as np

class AttrDict(dict):
    """A custom dict which converts dict keys
    to class attributes"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_mask_from_lengths(lengths, max_len=None, device='cuda'):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item().to(device)
    ids = torch.arange(0, max_len, requires_grad=False).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.to(device).unsqueeze(1).expand(-1, max_len)
    return mask.to(device)

def load_speaker_encoder_config(config_path):
    """Load config files and discard comments
    Args:
        config_path (str): path to config file.
    """
    config = AttrDict()

    ext = os.path.splitext(config_path)[1]
    if ext in (".yml", ".yaml"):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
    else:
        # fallback to json
        with open(config_path, "r") as f:
            input_str = f.read()
        # handle comments
        input_str = re.sub(r'\\\n', '', input_str)
        input_str = re.sub(r'//.*\n', '\n', input_str)
        data = json.loads(input_str)

    config.update(data)
    return config


def pad_1D(inputs, PAD=0):
# Len
    def pad_data(x, length, PAD):
        x_padded = F.pad(
            x, (0, length - x.shape[0]), mode="constant", value=PAD
        )
        return x_padded
    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])
    return padded

def pad_2D(inputs, maxlen=None):
    # List of tensors, dimension agnostic
    def _pad(x, max_len):
        PAD = 0
        if x.size(-1) > max_len:
            raise ValueError("not max_len")
        x_padded = F.pad(
            x, (0, max_len - x.size(-1)), mode="constant", value=PAD
        )
        return x_padded
    # [L, D] -> [D, L]
    reshaped = False
    if len(set([input.size(-1) for input in inputs])) == 1:
        inputs = [i.transpose(0,1) for i in inputs]
        reshaped = True
    if maxlen:
        output = torch.stack([_pad(x, maxlen) for x in inputs])
    else:
            max_len = max([x.shape[-1] for x in inputs])
            output = torch.stack([_pad(x, max_len) for x in inputs])
    if reshaped:
        output = torch.stack([o.transpose(0,1) for o in output])
    return output


def pad(inputs, max_len=None):
    # Expects list of tensors
    # 1D Padding
    if len(inputs[0].size()) == 1:
        return pad_1D(inputs)
    # 2D Padding
    elif len(inputs[0].size()) == 2:
        return pad_2D(inputs)
    else: ValueError('Expected [BS, N, L] or [BS, L]')  
    
    
