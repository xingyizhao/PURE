import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch import nn as nn
import torch.nn.functional as F


def set_seed(random_seed=11):
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

        cls_rep = sequence_output[:, 0, :]
        cls_output = self.dropout(cls_rep)

        return self.classifier(cls_output), attention_matrix


class BertLayerWiseClassification(nn.Module):
    def __init__(self, bert):
        super(BertLayerWiseClassification, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           output_hidden_states=True,
                           output_attentions=True)

        sequence_output = output[0]
        hidden_states = output[2]
        attention_matrix = output[3]

        output_logit_list = []

        for index_layer, hidden_layer in enumerate(hidden_states):
            # layer-wise output -- > hidden states + linear classifier
            if index_layer == 0:
                continue  # skip the embedding layer
            hidden_rep = hidden_layer[:, 0, :]
            output = self.dropout(hidden_rep)
            output_logit_list.append(self.classifier(output))

        # add the output of the last layer
        cls_rep = sequence_output[:, 0, :]
        output = self.dropout(cls_rep)
        output_logit_list.append(self.classifier(output))

        return output_logit_list, attention_matrix


# Define the loss function
class ERMLoss(nn.Module):
    def __init__(self, weight=None):
        super(ERMLoss, self).__init__()
        self.weight = weight

    def forward(self, y_pred, y_true):
        return F.cross_entropy(y_pred, y_true, weight=self.weight)


# Define the maximum entropy loss function
class MaxEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaxEntropyLoss, self).__init__()
        self.weight = weight
        self.softmax = nn.Softmax(dim=1)

    def forward(self, y_pred):
        self.weight = None
        y_pred = self.softmax(y_pred)
        return torch.sum(y_pred * torch.log(y_pred), dim=1).sum()  # log likelihood


# Define the attention regularization loss function
class AttentionRegularizationLoss(nn.Module):
    def __init__(self, penalty_coefficient, norm_mode="l2"):
        super(AttentionRegularizationLoss, self).__init__()
        self.penalty_coefficient = penalty_coefficient
        self.norm_mode = norm_mode

    def forward(self, attention_scores, cls_layer, coefficient_list):
        if self.norm_mode == "l2":
            regularization_term = 0.0
            for cls_index, coefficient in zip(cls_layer, coefficient_list):
                layer_attention_scores = attention_scores[cls_index]
                cls_attention_weights = layer_attention_scores[:, :, 0, :]

                square_tensor = cls_attention_weights * cls_attention_weights
                summed_tensor = square_tensor.sum(dim=2)
                tamp_tensor = summed_tensor.sum(dim=0)
                norm_l2 = torch.sum(torch.sqrt(tamp_tensor))
                regularization_term += self.penalty_coefficient * norm_l2 * coefficient
        else:
            raise ValueError("norm_mode: L2")

        return regularization_term
