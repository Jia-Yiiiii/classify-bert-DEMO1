import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import swanlab
import torch
import torch.nn as nn
import random
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

config_path = "./config/config.json"
with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)


class Config:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config(config_dict)


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

train_texts, train_labels = load_data(config.train_path)
dev_texts, dev_labels = load_data(config.dev_path)
test_texts, test_labels = load_data(config.test_path)

all_labels = sorted(list(set(train_labels + dev_labels + test_labels)))
label2id = {lab: i for i, lab in enumerate(all_labels)}
id2label = {i: lab for i, lab in enumerate(all_labels)}

train_labels = [label2id[lab] for lab in train_labels]
dev_labels = [label2id[lab] for lab in dev_labels]
test_labels = [label2id[lab] for lab in test_labels]


tokenizer = BertTokenizer.from_pretrained(config.model_name)

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = tokenizer(
            text,
            max_length=config.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_loader = DataLoader(NewsDataset(train_texts, train_labels), batch_size=config.batch_size, shuffle=True)
dev_loader = DataLoader(NewsDataset(dev_texts, dev_labels), batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(NewsDataset(test_texts, test_labels), batch_size=config.batch_size, shuffle=False)

class BertWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask=attention_mask)
        x = self.dropout(out.pooler_output)
        logits = self.classifier(x)
        return type('Out', (), {'logits': logits})()

model = BertWithDropout().to(config.device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay
)
total_steps = len(train_loader) * config.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
criterion = nn.CrossEntropyLoss()

def train_epoch(model, loader):
    model.train()
    total_loss, preds, labels = 0, [], []
    for batch in tqdm(loader, desc="Train"):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        label = batch["labels"].to(config.device)

        optimizer.zero_grad()
        out = model(input_ids, attention_mask)
        loss = criterion(out.logits, label)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
        labels.extend(label.cpu().numpy())
    return total_loss / len(loader), accuracy_score(labels, preds)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss, preds, labels = 0, [], []
    for batch in tqdm(loader, desc="Eval"):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        label = batch["labels"].to(config.device)

        out = model(input_ids, attention_mask)
        loss = criterion(out.logits, label)
        total_loss += loss.item()
        preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
        labels.extend(label.cpu().numpy())
    return total_loss / len(loader), accuracy_score(labels, preds), preds, labels


swanlab.init(project="bert-news-83+", config=config_dict)
best_acc = 0.0  

for epoch in range(config.epochs):
    print(f"\n===== Epoch {epoch + 1} =====")
    train_loss, train_acc = train_epoch(model, train_loader)
    dev_loss, dev_acc, _, _ = eval_epoch(model, dev_loader)
    if dev_acc > best_acc:
        best_acc = dev_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("已保存最优模型")

    swanlab.log({
        "train/loss": train_loss,
        "train/acc": train_acc,
        "dev/loss": dev_loss,
        "dev/acc": dev_acc
    })
    print(f"Train loss {train_loss:.4f} acc {train_acc:.4f}")
    print(f"Dev loss {dev_loss:.4f} acc {dev_acc:.4f}")

model.load_state_dict(torch.load("best_model.pth"))
test_loss, test_acc, test_pred, test_label = eval_epoch(model, test_loader)

print(f"测试集最终准确率：{test_acc:.4f}")
print("="*30)

swanlab.log({"test/acc": test_acc})
swanlab.finish()
