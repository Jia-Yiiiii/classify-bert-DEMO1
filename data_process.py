from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, file_path):
        self.texts = []
        self.labels = []
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
