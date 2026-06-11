import torch
import json
import swanlab
from transformers import BertTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from data_process import Load_Demo1_Data, get_textslabels, Myloader
from model import BertWithDropout
from utils import set_seed


class Trainer:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.num_epochs = config["epochs"]
        self.best_accuracy = 0.0
        self.counter = 0
        self.loss = torch.nn.CrossEntropyLoss()
        self.model = None
        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None
        self.label_id = None
        self.id_label = None
        set_seed(config.get("seed", 42))
    def load_data(self):
        train_data = Load_Demo1_Data(self.config["data_path"])
        dev_data = Load_Demo1_Data(self.config["dev_path"])
        test_data = Load_Demo1_Data(self.config["test_path"])
        unique_labels = []
        for item in train_data:
            label = item[1]
            if label not in unique_labels:
                unique_labels.append(label)
        self.label_id = {}
        self.id_label = {}
        for i in range(len(unique_labels)):
            self.label_id[unique_labels[i]] = i
            self.id_label[i] = unique_labels[i]
        self.train_texts, self.train_labels = get_textslabels(train_data, self.label_id)
        self.dev_texts, self.dev_labels = get_textslabels(dev_data, self.label_id)
        self.test_texts, self.test_labels = get_textslabels(test_data, self.label_id)

    def dataloader(self):

        tokenizer = BertTokenizer.from_pretrained(self.config["model_name"])

        self.train_loader = Myloader(
            self.train_texts, self.train_labels, tokenizer,
            self.config["batch_size"], self.config["max_len"], shuffle=True
        )
        self.dev_loader = Myloader(
            self.dev_texts, self.dev_labels, tokenizer,
            self.config["batch_size"], self.config["max_len"], shuffle=False
        )
        self.test_loader = Myloader(
            self.test_texts, self.test_labels, tokenizer,
            self.config["batch_size"], self.config["max_len"], shuffle=False
        )

    def init_model(self):
        num_classes = len(self.id_label)
        self.model = BertWithDropout(
            model_name=self.config["model_name"],
            dropout_rate=self.config.get("dropout_rate", 0.1),
            num_classes=num_classes
        )
        self.model.to(self.device)

    def savemodel(self):

        torch.save(self.model.state_dict(), "best_model.pth")

        with open("training_config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

        with open("DATA/label_id.txt", "w", encoding="utf-8") as f:
            for k, v in self.label_id.items():
                f.write(f"{k}:{v}\n")
        with open("DATA/id_label.txt", "w", encoding="utf-8") as f:
            for k, v in self.id_label.items():
                f.write(f"{k}:{v}\n")
        tokenizer = BertTokenizer.from_pretrained(self.config["model_name"])
        tokenizer.save_pretrained("tokenizer")

    def train(self):
        swanlab.init(
            project="bert-news-classification",
            config={
                "learning_rate": self.config["learning_rate"],
                "batch_size": self.config["batch_size"],
                "epochs": self.config["epochs"],
                "model": self.config["model_name"],
                "dropout_rate": self.config.get("dropout_rate", 0.1)
            }
        )

        optimizer = AdamW(self.model.parameters(), lr=self.config["learning_rate"])

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0

            for batch in self.train_loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
            avg_train_loss = total_loss / batch_count
            avg_eval_loss, eval_accuracy = self.eval()
            print("第", epoch + 1, "轮 训练loss:", avg_train_loss)
            print("验证准确率:", eval_accuracy)
            swanlab.log({
                "train/loss": avg_train_loss,
                "eval/loss": avg_eval_loss,
                "eval/acc": eval_accuracy,
                "epoch": epoch + 1
            })

            if eval_accuracy > self.best_accuracy:
                self.best_accuracy = eval_accuracy
                self.counter = 0
                self.savemodel()
                print("最佳模型的验证准确率:", eval_accuracy)
            else:
                self.counter += 1
            if self.counter >= self.config["patience"]:
                print("第", epoch + 1, "停止")
                break
        print("最好的验证准确率:", self.best_accuracy)
        self.test()
        swanlab.finish()

    def eval(self):
        self.model.eval()
        total_loss = 0
        batch_count = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in self.dev_loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss(logits, labels)
                total_loss += loss.item()
                batch_count += 1
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
        avg_loss = total_loss / batch_count
        accuracy = accuracy_score(all_labels, all_preds)
        return avg_loss, accuracy

    def test(self):
        self.model.load_state_dict(torch.load("best_model.pth", map_location=self.device))
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
        test_acc = accuracy_score(all_labels, all_preds)
        print("测试集准确率:", test_acc)
        swanlab.log({"test/acc": test_acc})


        self.plot(all_labels, all_preds)

    def plot(self, truths, preds):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        name_map = {
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
        for i in range(len(self.id_label)):
            en_name = self.id_label[i]
            cn_name = name_map.get(en_name, en_name)
            class_names.append(cn_name)

        print("\n分类报告:")
        print(classification_report(truths, preds, target_names=class_names))


        f1s = []
        for i in range(len(class_names)):
            tp = 0
            fp = 0
            fn = 0
            for j in range(len(truths)):
                if truths[j] == i and preds[j] == i:
                    tp += 1
                if truths[j] != i and preds[j] == i:
                    fp += 1
                if truths[j] == i and preds[j] != i:
                    fn += 1

            if tp + fp <= 0 or tp + fn <= 0:
                f1 = 0
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
        x = confusion_matrix(truths, preds)
        plt.figure(figsize=(12, 10))
        plt.imshow(x, cmap='Blues')
        plt.colorbar()
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, x[i, j], ha="center", va="center")

        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.show()

    def run(self):
        self.load_data()
        self.dataloader()
        self.init_model()
        self.train()


if __name__ == "__main__":
    with open("configs/Bert_Config_exp1.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    trainer = Trainer(config)
    trainer.run()
