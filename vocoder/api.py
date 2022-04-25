import json
import torch
from . import AttrDict, Generator


def get_vocoder(device):
    with open("vocoder/config.json", "r") as f:
        config = json.load(f)
    config = AttrDict(config)
    vocoder = Generator(config)
    ckpt = torch.load("vocoder/generator_universal.pth.tar",
                      map_location=device)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder.to(device)


def vocoder_infer(mels, vocoder, max_wav_value=32768.0, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels)
    wavs = (
        wavs.cpu().numpy() * max_wav_value
    ).astype("int16")
    wavs = [wav for wav in wavs]

    if lengths is not None:
        for i in range(len(mels)):
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
