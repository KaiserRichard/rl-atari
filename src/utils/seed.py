'''
Utility: Reproducibility

Reinforcement Leraning has randomness from:
- Neural network initialization
- Action sampling
- Environment randomness

Setting seeds allows reproducible experiences
'''

import torch
import random
import numpy as np

def set_seed(seed: int | None):
    '''
    Set all major random seeds

    Parameters:
        seed (int or None)
    
    If seed is None: 
        Do nothing
    '''

    # Python random
    random.seed(seed)

    # Pytorch random
    torch.manual_seed(seed)

    # Numpy random
    np.random.seed(seed)

    # GPU reproducibility (Locking down the math)
    if torch.cuda.is_available():
        # Force all GPUs to use the same starting seed for random operations
        torch.cuda.manual_seed_all(seed)

        # Force NVIDIA's CuDNN to use strictly predictable math
        # preventing chaotic rounding errors from parallel workers
        torch.backends.cudnn.deterministic = True

        # Stop PyTorch from auto-tuning and secretly swapping
        # convolutional algorithms in the background to save time
        torch.backends.cudnn.benchmark = False