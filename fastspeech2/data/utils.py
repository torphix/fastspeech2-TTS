import torch
from torch._C import device
import torch.nn.functional as F

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
    
    
def pad_to(inputs, target, mode='constant', PAD=0):
    # Pads input to match target
    # Inputs: [BS, N, L]
    return F.pad(inputs, (0, target.size(-1) - inputs.size(-1)), mode='constant', value=PAD)


