import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import swanlab
from data_process import load_data, build_label_map, init_tokenizer, NewsDataset
from model import BertWithDropout
from utils import set_seed, train_epoch, eval_epoch

set_seed()


config_path = "./config_exp/config.json"
with open(config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)

class Config:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config(config_dict)


train_texts, train_labels = load_data(config.train_path)
dev_texts, dev_labels = load_data(config.dev_path)
test_texts, test_labels = load_data(config.test_path)

label2id, id2label = build_label_map(train_labels, dev_labels, test_labels)

train_labels = [label2id[lab] for lab in train_labels]
dev_labels = [label2id[lab] for lab in dev_labels]
test_labels = [label2id[lab] for lab in test_labels]


init_tokenizer(config.model_name)


train_dataset = NewsDataset(train_texts, train_labels, config.max_len)
dev_dataset = NewsDataset(dev_texts, dev_labels, config.max_len)
test_dataset = NewsDataset(test_texts, test_labels, config.max_len)


train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)


model = BertWithDropout(
    model_name=config.model_name,
    dropout_rate=config.dropout_rate,
    num_classes=config.num_classes
).to(config.device)

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


swanlab.init(project="bert-news-83+", config=config_dict)
best_acc = 0.0

for epoch in range(config.epochs):
    print(f"\n===== Epoch {epoch + 1} =====")

    
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler, criterion, config.device
    )

  
    dev_loss, dev_acc, _, _ = eval_epoch(model, dev_loader, criterion, config.device)

  
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
test_loss, test_acc, test_pred, test_label = eval_epoch(model, test_loader, criterion, config.device)

print(f"\n测试集最终准确率：{test_acc:.4f}")
swanlab.log({"test/acc": test_acc})
swanlab.finish()
