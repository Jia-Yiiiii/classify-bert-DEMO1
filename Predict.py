import torch
from transformers import BertTokenizer, BertForSequenceClassification
from utils import load_config

config = load_config("configs/Bert_Config_exp1.json")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open("DATA/id_label.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

id_label = {}
for line in lines:
    line = line.strip()
    if line:
        parts = line.split(':')
        id_label[int(parts[0])] = parts[1]

model = BertForSequenceClassification.from_pretrained(
    config["model_name"],
    num_labels=len(id_label)
)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained(config["model_name"])


def predict(text):
    x = tokenizer(text, truncation=True, padding='max_length',
                  max_length=config["max_len"], return_tensors='pt')
    input_ids = x['input_ids'].to(device)
    mask = x['attention_mask'].to(device)

    with torch.no_grad():
        out = model(input_ids, mask)
        pred_id = torch.argmax(out.logits, dim=1).item()
    return id_label[pred_id]


if __name__ == "__main__":
    text = input("请输入: ")
    if text.strip():
        result = predict(text)
        print("预测类别:", result)
    else:
        print("输入不能为空")
