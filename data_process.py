import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


tokenizer = None

def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('_!_')
            if len(parts) >= 4:
                texts.append(parts[3])
                labels.append(parts[2])
    return texts, labels

def build_label_map(train_labels, dev_labels, test_labels):
    all_labels = sorted(list(set(train_labels + dev_labels + test_labels)))
    label2id = {lab: i for i, lab in enumerate(all_labels)}
    id2label = {i: lab for i, lab in enumerate(all_labels)}
    return label2id, id2label

def init_tokenizer(model_name):
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, max_len):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
