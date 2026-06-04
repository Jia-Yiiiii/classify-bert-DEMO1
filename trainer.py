import torch
from torch.utils.data import DataLoader
import swanlab
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_process import Demo1_Dataset, collate_fn,Load_Demo1_Data,get_textslabels,Myloader,config
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

swanlab.init(
    project="bert-news-classification",
    config={
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "model": config["model_name"]
    }
)

data = Load_Demo1_Data(config["data_path"])
dev_data = Load_Demo1_Data(config["dev_path"])
test_data = Load_Demo1_Data(config["test_path"])


lables = []
for i in data:
    lables.append(i[1])

unique_labels = []
for i in lables:
    if i not in unique_labels:
        unique_labels.append(i)

label_id = {}
id_label = {}
for i in range(len(unique_labels)):
    label_id[unique_labels[i]] = i
    id_label[i] = unique_labels[i]


texts, numlabels = get_textslabels(data, label_id)
dev_texts, dev_labels = get_textslabels(dev_data, label_id)
test_texts, test_labels = get_textslabels(test_data, label_id)



tokenizer = BertTokenizer.from_pretrained(config["model_name"])


train_loader = Myloader(texts, numlabels, tokenizer, config["batch_size"], shuffle=True)
dev_loader = Myloader(dev_texts, dev_labels, tokenizer, config["batch_size"], shuffle=False)
test_loader = Myloader(test_texts, test_labels, tokenizer, config["batch_size"], shuffle=False)



num_classes = len(unique_labels)
model = BertForSequenceClassification.from_pretrained(config["model_name"], num_labels=num_classes)
model.to(device)


optimizer = AdamW(model.parameters(), lr=config["learning_rate"])


epochs = config["epochs"]
best_acc = 0.0
counter = 0
stop = config["patience"]

for epoch in range(epochs):
    model.train()
    total_loss = 0
    batch_count = 0

    for batch in train_loader:
        inputid = batch[0].to(device)
        masks = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()
        outputs = model(inputid, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / batch_count
    print("第", epoch+1, "轮 训练loss:", avg_loss)


    model.eval()
    p = []
    label = []

    with torch.no_grad():
        for batch in dev_loader:
            inputid = batch[0].to(device)
            masks = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(inputid, attention_mask=masks)
            preds = torch.argmax(outputs.logits, dim=1)

            p.extend(preds.numpy())
            label.extend(labels.numpy())

    val_acc = accuracy_score(label, p)
    print("验证准确率:", val_acc)

    swanlab.log({
        "train/loss": avg_loss,
        "val/acc": val_acc,
        "epoch": epoch + 1
    })

    if val_acc > best_acc:
        best_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("验证准确率:", val_acc)
    else:
        counter += 1
    if counter >= stop:
        break

print("最好的验证准确率", best_acc)



model.load_state_dict(torch.load("best_model.pth"))
model.eval()
test_preds = []
test_true = []

with torch.no_grad():
    for batch in test_loader:
        inputid = batch[0].to(device)
        masks = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(inputid, attention_mask=masks)
        preds = torch.argmax(outputs.logits, dim=1)

        test_preds.extend(preds.numpy())
        test_true.extend(labels.numpy())

test_acc = accuracy_score(test_true, test_preds)
print("测试集准确率:", test_acc)

map = {
    "news_story": "新闻故事",
    "news_culture": "新闻文化",
    "news_entertainment": "新闻娱乐",
    "news_sports": "新闻体育",
    "news_finance": "新闻财经",
    "news_house": "新闻房产",
    "news_car": "新闻汽车",
    "news_edu": "新闻教育",
    "news_tech": "新闻科技",
    "news_military": "新闻军事",
    "news_travel": "新闻旅游",
    "news_world": "新闻国际",
    "stock": "股票",
    "news_agriculture": "新闻农业",
    "news_game": "新闻游戏",
}

class_names = []
for i in range(len(id_label)):
    en_name = id_label[i]
    cn_name = map.get(en_name, en_name)
    class_names.append(cn_name)

print("\n分类报告:")
print(classification_report(test_true, test_preds, target_names=class_names))


f1s = []
for i in range(len(class_names)):
    tp = 0
    fp = 0
    fn = 0
    for j in range(len(test_true)):
        if test_true[j] == i and test_preds[j] == i:
            tp += 1
        elif test_true[j] != i and test_preds[j] == i:
            fp += 1
        elif test_true[j] == i and test_preds[j] != i:
            fn += 1
    if tp + fp == 0 or tp + fn == 0:
        f1s.append(0)
    else:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        f1s.append(f1)

plt.figure(figsize=(12, 5))
plt.bar(class_names, f1s)
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.ylabel('F1')
plt.title('各类别F1分数')
plt.tight_layout()
plt.show()


cm = confusion_matrix(test_true, test_preds)
plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.title('混淆矩阵')
plt.tight_layout()
plt.show()
swanlab.log({"test/acc": test_acc})


with open("DATA/label_id.txt", "w", encoding="utf-8") as f:
    for k, v in label_id.items():
        f.write(f"{k}:{v}\n")

with open("DATA/id_label.txt", "w", encoding="utf-8") as f:
    for k, v in id_label.items():
        f.write(f"{k}:{v}\n")

swanlab.finish()
