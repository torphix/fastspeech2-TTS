import os
import re
import json


class AttrDict(dict):
    """A custom dict which converts dict keys
    to class attributes"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(config_path: str) -> AttrDict:
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