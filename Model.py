import torch
import numpy as np
from transformers import BertTokenizer
import pandas as pd
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
import swanlab


swanlab.init(
    project="news-classify-bert",
    config={
        "data_num": 30000,
        "epochs": 8,
        "lr": 2e-5,
        "batch_size": 8,
        "max_len": 128,
        "dropout": 0.3,
        "device": "cpu"
    }
)


df = pd.read_csv('DATA/toutiao_cat_data.csv')
df = df.head(30000)

np.random.seed(42)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                     [int(.8 * len(df)), int(.9 * len(df))])

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels[label] for label in df['label']]
        self.texts = [tokenizer(str(text),
                                padding='max_length',
                                max_length=128,
                                truncation=True,
                                return_tensors="pt") for text in df['text']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('./bert-base-chinese')
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, 15)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        return self.linear(self.dropout(pooled_output))


def train(model, train_data, val_data, lr, epochs, batch_size):
    train_loader = torch.utils.data.DataLoader(Dataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(Dataset(val_data), batch_size=batch_size)

    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        for x, y in tqdm(train_loader):
            y = y.to(device)
            out = model(
                input_ids=x['input_ids'].squeeze(1).to(device),
                attention_mask=x['attention_mask'].squeeze(1).to(device)
            )

            loss = criterion(out, y)
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = correct / len(train_data)
        train_loss = total_loss / len(train_loader)

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                y = y.to(device)
                out = model(
                    input_ids=x['input_ids'].squeeze(1).to(device),
                    attention_mask=x['attention_mask'].squeeze(1).to(device)
                )
                val_correct += (out.argmax(1) == y).sum().item()
        val_acc = val_correct / len(val_data)


        swanlab.log({
            "Train Acc": train_acc,
            "Val Acc": val_acc,
            "Train Loss": train_loss
        }, step=epoch + 1)

        print(f"Epoch {epoch + 1} | "
              f"Train Acc: {train_acc:.3f} | "
              f"Val Acc: {val_acc:.3f}")


model = BertClassifier()


train(model, df_train, df_val, lr=2e-5, epochs=8, batch_size=8)


def test(model, test_data):
    loader = torch.utils.data.DataLoader(Dataset(test_data), batch_size=8)
    device = torch.device("cpu")
    model.to(device)
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            y = y.to(device)
            out = model(input_ids=x['input_ids'].squeeze(1).to(device),
                        attention_mask=x['attention_mask'].squeeze(1).to(device))
            correct += (out.argmax(1) == y).sum().item()
    test_acc = correct / len(test_data)
    print(f"\nTest Accuracy: {test_acc:.3f}")

    swanlab.log({"Test Acc": test_acc})


test(model, df_test)
swanlab.finish()
