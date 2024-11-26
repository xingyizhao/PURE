from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from util import *
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from transformers import get_linear_schedule_with_warmup
import math
import json
from config import get_arguments
from tqdm import tqdm
import datasets


"""
PURE -- Step 2: Attention Head Normalization (Defender): [Random Seed of 5 runs: 11, 12, 13, 14, 15]
1. Loading the Pruned Heads information from head_coefficients folder.
2. Fine-tuning the pruned model using the AttentionRegularizationLoss in util.py. 

-----------------
This file contains the code of the PURE algorithm for attention head pruning.

- Implementation details follow the default settings in the code.
- Attacker poisoning process is implemented on the IMDB, YELP and SST-2 dataset. 
- Puring process is only implemented with the SST-2 dataset.
- I save the pruned head information and coefficients in the head_coefficients folder.
- Author email: xingyi.zhao@usu.edu   
-----------------
"""


def train(model, dataloader, optimizer, loss_fn, scheduler, attention_loss, cls_layer, coefficients):
    total_loss = 0

    for index, batch in enumerate(dataloader):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        logits, attention_matrix = model(input_ids=input_ids, attention_mask=attention_mask)

        # Equation 3 in the paper
        loss = loss_fn(logits, targets) + attention_loss(attention_matrix, cls_layer, coefficients)
        total_loss += loss.item()

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


def evaluate_on_test(model, dataloader):
    preds = []
    all_targets = []

    with torch.no_grad():
        for index, batch in tqdm(enumerate(dataloader)):
            model.eval()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"]

            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
            preds.extend(predictions)
            all_targets.extend(targets.numpy())

    accuracy = accuracy_score(all_targets, preds)  # Accuracy on the validation set
    cm = confusion_matrix(all_targets, preds)  # Confusion matrix on the validation set

    return accuracy, cm


def coefficient_normalization_variance(coefficient_list):
    # use -log function to normalize the coefficients
    coefficient_list = [-math.log(x) for x in coefficient_list]

    # use softmax function to normalize the coefficients
    coefficient_list = [math.exp(x) for x in coefficient_list]
    coefficient_list = [x / sum(coefficient_list) for x in coefficient_list]

    return coefficient_list


def load_coefficients_from_txt(filename="head_coefficients/norm_coefficients.txt"):
    with open(filename, 'r') as infile:
        return [float(line.rstrip()) for line in infile]


def load_from_file(filename="prune_heads.json"):
    with open(filename, 'r') as infile:
        loaded_data = json.load(infile)
        return {int(key): value for key, value in loaded_data.items()}


if __name__ == '__main__':
    args = get_arguments().parse_args()

    set_seed(random_seed=args.seed)

    device = torch.device(args.device)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    # Load the target dataset
    sst = datasets.load_dataset("sst2")
    sst_train = sst["train"].train_test_split(train_size=0.9, seed=11)  # 90% for training, 10% for validation

    clean_train_df = pd.DataFrame({"label": sst_train["train"]["label"], "text": sst_train["train"]["sentence"]})

    # To check the defense performance, we use the clean test and poisoned test provided by the attacker
    # The key idea is to check whether the defense method can recover the performance of the poisoned dataset
    clean_test_df = pd.read_csv(f"../Clean Data/Attacker-SST-2/clean_test.csv")
    if args.attack_mode == "BadNet" or args.attack_mode == "LayerWise":
        poisoned_test_df = pd.read_csv(f"../Data/Rare_Word/SST-2/poisoned_test.csv")
    else:
        poisoned_test_df = pd.read_csv(f"../Data/{args.attack_mode}/SST-2/poisoned_test.csv")

    clean_train_dataset = TargetDataset(tokenizer=tokenizer, max_len=args.max_len_short, data=clean_train_df)
    clean_test_dataset = TargetDataset(tokenizer=tokenizer, max_len=args.max_len_short, data=clean_test_df)
    poisoned_test_dataset = TargetDataset(tokenizer=tokenizer, max_len=args.max_len_short, data=poisoned_test_df)

    clean_train_loader = DataLoader(clean_train_dataset, batch_size=args.batch_size, shuffle=False)
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=args.batch_size, shuffle=False)
    poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load the pruned heads information
    Heads = load_from_file("head_coefficients/pruned_heads.json")

    # Load the coefficients and normalize them
    cls_layer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    coefficients = load_coefficients_from_txt("head_coefficients/norm_coefficients.txt")
    coefficients = coefficient_normalization_variance(coefficients)
    normalized_coefficients = (coefficients - np.min(coefficients)) / (np.max(coefficients) - np.min(coefficients))
    print(normalized_coefficients)

    # Prune Step 1: Prune the attention heads
    pured_bert = BertModel.from_pretrained(f"poisoned_model/{args.attack_mode}")
    pured_model = BertClassification(pured_bert).to(device)
    pured_model.bert.prune_heads(Heads)

    optimizer = torch.optim.AdamW(pured_model.parameters(), lr=args.learning_rate)
    total_steps = len(clean_train_loader) * args.defending_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # define the loss function
    loss_fn = ERMLoss()
    attention_loss = AttentionRegularizationLoss(penalty_coefficient=args.penalty_coefficient, norm_mode="l2")

    # PURE Step 2: Fine-tuning the pruned model combining the cross-entropy loss and the attention regularization loss
    for epoch in tqdm(range(args.defending_epoch)):
        train_loss = train(pured_model,
                           clean_train_loader,
                           optimizer,
                           loss_fn,
                           scheduler,
                           attention_loss,
                           cls_layer,
                           normalized_coefficients)
        print(f"Epoch: {epoch + 1} / {args.defending_epoch}, | Train Loss: {train_loss}")

    # Evaluate the model on both clean and poisoned test sets
    clean_accuracy, clean_cm = evaluate_on_test(pured_model, clean_test_loader)
    poisoned_accuracy, poisoned_cm = evaluate_on_test(pured_model, poisoned_test_loader)

    lfr_negative_clean = clean_cm[0][1] / (clean_cm[0][0] + clean_cm[0][1])
    lfr_negative_poisoned = poisoned_cm[0][1] / (poisoned_cm[0][0] + poisoned_cm[0][1])

    print("Results on the clean test set")
    print(f"Clean Accuracy:\n {clean_accuracy}")
    print(f"Clean Confusion Matrix:\n {clean_cm}")
    print(f"LFR Negative Clean: {lfr_negative_clean}")

    print("Results on the poisoned test set")
    print(f"Poisoned Accuracy:\n {poisoned_accuracy}")
    print(f"Poisoned Confusion Matrix:\n {poisoned_cm}")
    print(f"LFR Negative Poisoned: {lfr_negative_poisoned}")
