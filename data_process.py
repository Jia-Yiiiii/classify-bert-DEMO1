import torch
from torch.utils.data import Dataset, DataLoader


class Demo1_Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def collate_fn(self, batch):
        texts = []
        labels = []
        for i in range(len(batch)):
            texts.append(batch[i][0])
            labels.append(batch[i][1])

        x = self.tokenizer(texts, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return x['input_ids'], x['attention_mask'], torch.tensor(labels)


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


def Myloader(texts, labels, tokenizer, batch_size, max_len, shuffle=True):
    dataset = Demo1_Dataset(texts, labels, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn)
