import torch
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, file_path, tokenizer=None, config=None):
        self.texts = []
        self.labels = []
        self.label2id = {}  
        self.tokenizer = tokenizer
        self.config = config

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('_!_')
                if len(parts) >= 4:
                    self.texts.append(parts[3])
                    self.labels.append(parts[2])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def collate_fn(self, batch):
        texts = [item[0] for item in batch]
        labels = [self.label2id[item[1]] for item in batch]
        
        enc = self.tokenizer(
            texts,
            max_length=self.config.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long)
        }
