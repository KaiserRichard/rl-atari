''''
Helper functions for tensor handling
'''

import torch

def to_tensor(array, device, dtype=torch.float32):
    '''
    Convert NumPy array to torch tensor on correct device
    '''

    return torch.as_tensor(array,dtype=dtype,device=device)


def add_batch_dim(tensor):
    '''
    Add batch dimension because Neural Networks expect batch dimension
    '''
    return tensor.unsqueeze(0)
