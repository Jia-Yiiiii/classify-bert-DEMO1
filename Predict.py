import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import json
import os

from utils import load_config
from model import BertWithDropout
from data_process import NewsDataset


config, config_dict = load_config("./configs/Bert_Config_exp1.json")
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("./DATA/label2id.json", "r", encoding="utf-8") as f:
    label2id = json.load(f)
with open("./DATA/id2label.json", "r", encoding="utf-8") as f:
    id2label = json.load(f)

all_labels = list(label2id.keys())


tokenizer = BertTokenizer.from_pretrained(config.model_name)
model = BertWithDropout(config).to(config.device)
model.load_state_dict(torch.load("best_model.pth", map_location=config.device))
model.eval()


test_dataset = NewsDataset(
    file_path=config.test_path,
    tokenizer=tokenizer,
    config=config
)
test_dataset.label2id = label2id

test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=test_dataset.collate_fn
)


@torch.no_grad()
def get_preds():
    preds, trues, confs = [], [], []
    for batch in test_loader:
        input_ids = batch["input_ids"].to(config.device)
        att_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"]

        logits = model(input_ids=input_ids, attention_mask=att_mask)
        conf, pred = torch.max(torch.softmax(logits, dim=1), dim=1)

        preds.extend(pred.cpu().numpy())
        trues.extend(labels.numpy())
        confs.extend(conf.cpu().numpy())
    return trues, preds, confs


def predict_single(text):
    inputs = tokenizer(
        text,
        max_length=config.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(config.device)

    logits = model(**inputs)
    pred_id = torch.argmax(logits, dim=1).item()
    return id2label[str(pred_id)]


if __name__ == "__main__":
    trues, preds, confs = get_preds()

    demo_text = "神舟十八号载人飞船成功发射，圆满完成任务！"
    pred_label = predict_single(demo_text)
    print(f"输入文本：{demo_text}")
    print(f"预测类别：{pred_label}")
    print("=" * 50)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(trues, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels)
    plt.title('混淆矩阵')
    plt.xlabel('预测')
    plt.ylabel('真实')
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

    plt.figure(figsize=(12, 5))
    sns.barplot(x=all_labels, y=acc_per_class)
    plt.title('各类别准确率')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    count = np.bincount(trues, minlength=len(all_labels))
    plt.figure(figsize=(8, 8))
    plt.pie(count, labels=all_labels, autopct='%1.1f%%')
    plt.title('标签分布')
    plt.show()
    plt.figure(figsize=(10, 5))
    sns.histplot(confs, bins=30, kde=True, color='orange')
    plt.title('置信度分布')
    plt.show()


    acc = accuracy_score(trues, preds)
    p = precision_score(trues, preds, average='macro')
    r = recall_score(trues, preds, average='macro')
    f1 = f1_score(trues, preds, average='macro')

    print("\n模型评估指标")
    print(f"准确率: {acc:.4f}")
    print(f"精确率: {p:.4f}")
    print(f"召回率: {r:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print("\n详细报告：")
    print(classification_report(trues, preds, target_names=all_labels, digits=4))
