import random
import numpy as np
import torch
import json
from transformers import BertTokenizer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    class BertConfig:
        def __init__(self, config_dict):
            for k, v in config_dict.items():
                setattr(self, k, v)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return BertConfig(config_dict), config_dict

def collate_fn(batch, tokenizer, config):
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    enc = tokenizer(texts,max_length=config.max_len,padding="max_length",truncation=True,return_tensors="pt")
    return {"input_ids": enc["input_ids"],"attention_mask": enc["attention_mask"],"labels": torch.tensor(labels, dtype=torch.long)}
