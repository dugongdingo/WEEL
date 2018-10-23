import datetime

import torch

from .settings import DEVICE

def to_tensor(seq, device=DEVICE) :
    return torch.tensor(seq, dtype=torch.long, device=device).view(-1, 1)

def print_now(*line) :
    """
    utility function: prepend timestamp to std output
    """
    print(datetime.datetime.now(), ":", *line)
