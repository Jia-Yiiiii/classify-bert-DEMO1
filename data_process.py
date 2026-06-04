import torch
from torch.utils.data import Dataset, DataLoader
from utils import load_config
config = load_config("configs/Bert_Config_exp1.json")
class Demo1_Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        x = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
        )

        input_ids = torch.tensor(x['input_ids'])
        attention_mask = torch.tensor(x['attention_mask'])
        label = torch.tensor(label)
        return input_ids, attention_mask, label


def collate_fn(batch):
    all_ids = []
    all_masks = []
    all_labels = []

    for i in batch:
        all_ids.append(i[0])
        all_masks.append(i[1])
        all_labels.append(i[2])

    input_ids = torch.stack(all_ids, dim=0)
    masks = torch.stack(all_masks, dim=0)
    labels = torch.stack(all_labels, dim=0)

    return input_ids, masks, labels

def Load_Demo1_Data(file_path):

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('_!_')
            if len(parts) >= 5:
                title = parts[3]
                keywords = parts[4].strip()
                text = title + "，" + keywords
                label = parts[2]
                data.append((text, label))
    return data

def get_textslabels(data, label_id):
    texts = []
    labels = []
    for i in data:
        texts.append(i[0])
        labels.append(label_id[i[1]])
    return texts, labels

def Myloader(texts, labels, tokenizer, batch_size, shuffle=True):
    dataset = Demo1_Dataset(texts, labels, tokenizer, max_len=config["max_len"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

