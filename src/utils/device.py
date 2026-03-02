'''
Utility: Device Seletion 
'''

import torch

def get_device():
    '''
    Returns the best available device
    '''
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.is_avaible()
        else "cpu"
    )