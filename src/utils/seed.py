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

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True