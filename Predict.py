import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score  
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


class Config:
    model_name = "bert-base-chinese"
    test_path = "DATA/test_1k.txt"
    max_len = 100
    batch_size = 16
    num_classes = 15
    dropout_rate = 0.4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()


def load_data(file_path):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('_!_')
            if len(parts) >= 4:
                texts.append(parts[3])
                labels.append(parts[2])
    return texts, labels

test_texts, test_labels = load_data(config.test_path)
all_labels = sorted(list(set(test_labels)))
label2id = {l:i for i,l in enumerate(all_labels)}
id2label = {i:l for i,l in enumerate(all_labels)}
test_labels_id = [label2id[l] for l in test_labels]


class BertWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_classes)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(out.pooler_output)
        logits = self.classifier(x)
        return type('Out', (), {'logits': logits})()

tokenizer = BertTokenizer.from_pretrained(config.model_name)
model = BertWithDropout().to(config.device)
model.load_state_dict(torch.load("best_model.pth", map_location=config.device))
model.eval()


class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], max_length=config.max_len, padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].flatten(), "attention_mask": enc["attention_mask"].flatten(), "labels": torch.tensor(self.labels[idx])}

test_loader = DataLoader(NewsDataset(test_texts, test_labels_id), batch_size=config.batch_size, shuffle=False)


@torch.no_grad()
def get_preds():
    preds, trues, confs = [], [], []
    for batch in test_loader:
        input_ids = batch["input_ids"].to(config.device)
        att_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"]
        logits = model(input_ids, att_mask).logits
        conf, pred = torch.max(torch.softmax(logits, dim=1), dim=1)
        preds.extend(pred.cpu().numpy())
        trues.extend(labels.numpy())
        confs.extend(conf.cpu().numpy())
    return trues, preds, confs

trues, preds, confs = get_preds()


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


plt.figure(figsize=(12,10))
cm = confusion_matrix(trues, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels)
plt.title('混淆矩阵 Confusion Matrix', fontsize=16)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.show()

acc_per_class = []
for i in range(len(all_labels)):
    mask = np.array(trues) == i
    if np.sum(mask) == 0:
        acc_per_class.append(0)
    else:
        acc = accuracy_score(np.array(trues)[mask], np.array(preds)[mask])
        acc_per_class.append(acc)

plt.figure(figsize=(12,5))
sns.barplot(x=all_labels, y=acc_per_class, palette='viridis')
plt.title('各类别准确率', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylim(0,1)
plt.tight_layout()
plt.show()


labels_count = np.bincount(trues, minlength=len(all_labels))
plt.figure(figsize=(8,8))
plt.pie(labels_count, labels=all_labels, autopct='%1.1f%%', startangle=90)
plt.title('标签分布', fontsize=14)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,5))
sns.histplot(confs, bins=30, kde=True, color='orange')
plt.title('模型预测置信度分布', fontsize=14)
plt.xlabel('置信度')
plt.tight_layout()
plt.show()



accuracy = accuracy_score(trues, preds)
macro_precision = precision_score(trues, preds, average='macro')
macro_recall = recall_score(trues, preds, average='macro')
macro_f1 = f1_score(trues, preds, average='macro')

print("\n" + "="*60)
print("模型综合评估指标")
print("="*60)
print(f"总体准确率 Accuracy: {accuracy:.4f}")
print(f"宏平均精确率 Precision: {macro_precision:.4f}")
print(f"宏平均召回率 Recall: {macro_recall:.4f}")
print(f"宏平均F1分数 Macro-F1: {macro_f1:.4f}")
print("="*60)
print("\n详细分类报告：")
print(classification_report(trues, preds, target_names=all_labels, digits=4))
