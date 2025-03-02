# utils/helpers.py
import os
import random
import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def worker_init_fn(worker_id, seed=1):
    np.random.seed(seed + worker_id)

def compute_loss_weighting(sigmas, scheme='reciprocal'):
    if scheme == 'none':
        return torch.ones_like(sigmas)
    elif scheme == 'reciprocal':
        return 1.0 / (sigmas ** 2 + 1e-8)
    elif scheme == 'sigma_sqrt':
        return torch.sqrt(sigmas)
    else:
        return torch.ones_like(sigmas)
