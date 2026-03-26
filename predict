import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel


tokenizer = BertTokenizer.from_pretrained(r'D:\PythonProject3\BERT\bert-base-chinese')


labels = {
    'news_story': 0,
    'news_culture': 1,
    'news_entertainment': 2,
    'news_sports': 3,
    'news_finance': 4,
    'news_house': 5,
    'news_car': 6,
    'news_edu': 7,
    'news_tech': 8,
    'news_military': 9,
    'news_travel': 10,
    'news_world': 11,
    'stock': 12,
    'news_agriculture': 13,
    'news_game': 14
}
id2label = {v: k for k, v in labels.items()}

# 模型结构
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(r'D:\PythonProject3\BERT\bert-base-chinese')
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, 15)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        return self.linear(self.dropout(pooled_output))


model = BertClassifier()
model.eval()

# 预测函数
def predict_sentence(sentence):
    device = torch.device("cpu")
    encoded = tokenizer(
        sentence,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        out = model(
            input_ids=encoded['input_ids'].to(device),
            attention_mask=encoded['attention_mask'].to(device)
        )
    pred_class = out.argmax(1).item()
    pred_label = id2label[pred_class]
    print(f"\n输入：{sentence}")
    print(f"分类：{pred_label} [{pred_class}]")
    return pred_label

predict_sentence("中国女排3比0战胜美国队夺冠！")
predict_sentence("A股今天大涨，新能源板块领涨")
