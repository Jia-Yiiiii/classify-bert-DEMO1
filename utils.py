import json
import torch
import random
import numpy as np

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
