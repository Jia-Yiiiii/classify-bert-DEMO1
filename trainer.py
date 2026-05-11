import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import swanlab
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from utils import set_seed, load_config
from data_process import NewsDataset
from model import BertWithDropout

def train_epoch(model, loader, optimizer, scheduler, criterion, config):
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
def eval_epoch(model, loader, criterion, config):
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

def main():
    set_seed(42)
    config, config_dict = load_config("./configs/Bert_Config_exp1.json")
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    train_dataset = NewsDataset(config.train_path, tokenizer, config)
    dev_dataset = NewsDataset(config.dev_path, tokenizer, config)
    test_dataset = NewsDataset(config.test_path, tokenizer, config)


    all_labels = sorted({lab for _, lab in train_dataset} | {lab for _, lab in dev_dataset})
    label2id = {lab: i for i, lab in enumerate(all_labels)}
    train_dataset.label2id = label2id
    dev_dataset.label2id = label2id
    test_dataset.label2id = label2id

    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn 
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=dev_dataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )


    model = BertWithDropout(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(0.1 * total_steps),num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    swanlab.init(project="bert-news-83+", config=config_dict)
    best_acc = 0.0
    for epoch in range(config.epochs):
        print(f"\n===== Epoch {epoch + 1} =====")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, config)
        dev_loss, dev_acc, _, _ = eval_epoch(model, dev_loader, criterion, config)
        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("已保存最优模型")
        swanlab.log({"train/loss": train_loss,"train/acc": train_acc,"dev/loss": dev_loss,"dev/acc": dev_acc})
        print(f"Train loss {train_loss:.4f} acc {train_acc:.4f}")
        print(f"Dev loss {dev_loss:.4f} acc {dev_acc:.4f}")
    model.load_state_dict(torch.load("best_model.pth", map_location=config.device))
    model.eval()
    test_loss, test_acc, test_pred, test_label = eval_epoch(model, test_loader, criterion, config)
    print("\n" + "="*30)
    print(f"测试集最终准确率：{test_acc:.4f}")
    print("="*30)
    swanlab.log({"test/acc": test_acc})
    swanlab.finish()
if __name__ == "__main__":
    main()
