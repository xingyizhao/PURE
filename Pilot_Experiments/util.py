import random
import numpy as np
from torch import nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset


def set_seed(random_seed=11):
    # Set the seed value all over the place to make this reproducible.
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


class TargetDataset(Dataset):
    def __init__(self, tokenizer, max_len, data):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text = data["text"]
        self.targets = data["label"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


class BertClassification(nn.Module):
    def __init__(self, bert):
        super(BertClassification, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           output_hidden_states=True,
                           output_attentions=True)

        sequence_output = output[0]
        attention_matrix = output[3]

        output_logit_list = []
        cls_rep = sequence_output[:, 0, :]
        output = self.dropout(cls_rep)
        output_logit_list.append(self.classifier(output))

        return output_logit_list, attention_matrix


# Define the loss function [Empirical Risk Minimization (ERM) Loss]
class ERMLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(ERMLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        return F.cross_entropy(y_pred, y_true, weight=self.weight, reduction=self.reduction)
