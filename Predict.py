import torch
import json
from transformers import BertTokenizer
from model import BertWithDropout


class Predictor:
    def __init__(self, model_path="best_model.pth", config_path="training_config.json"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        with open("DATA/id_label.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.id_label = {}
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split(':')
                self.id_label[int(parts[0])] = parts[1]
        self.tokenizer = BertTokenizer.from_pretrained("tokenizer")
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        num_classes = len(self.id_label)
        model = BertWithDropout(
            model_name=self.config["model_name"],
            dropout_rate=self.config.get("dropout_rate", 0.1),
            num_classes=num_classes
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict(self, text):
        x = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config["max_len"],
            return_tensors='pt'
        )
        input_ids = x['input_ids'].to(self.device)
        mask = x['attention_mask'].to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=mask)
            pred_id = torch.argmax(logits, dim=1).item()
        return self.id_label[pred_id]


if __name__ == "__main__":
    predict = Predictor()
    text = input("请输入: ")
    if text.strip():
        result = predict.predict(text)
        print("预测类别:", result)
    else:
        print("输入不能为空")
