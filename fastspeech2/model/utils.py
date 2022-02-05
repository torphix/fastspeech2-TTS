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