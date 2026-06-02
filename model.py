import torch
import torch.nn as nn
from transformers import BertModel

class BertWithDropout(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_classes)
    def forward(self, **kwargs):
        out = self.bert(** kwargs)
        x = self.dropout(out.pooler_output)
        logits = self.classifier(x)
        return logits

