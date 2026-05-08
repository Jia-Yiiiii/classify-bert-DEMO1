import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, preds, labels = 0, [], []

    for batch in tqdm(loader, desc="Train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["labels"].to(device)

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
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0, [], []

    for batch in tqdm(loader, desc="Eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["labels"].to(device)

        out = model(input_ids, attention_mask)
        loss = criterion(out.logits, label)
        total_loss += loss.item()

        preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
        labels.extend(label.cpu().numpy())

    return total_loss / len(loader), accuracy_score(labels, preds), preds, labels
