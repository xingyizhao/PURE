from config import get_arguments
import pandas as pd
from util import *
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from tqdm import tqdm

"""
Poisoning Phase (Attacker):
-----------------
This file contains the code of Main Experiments of the paper "Defense against Backdoor Attack on Pre-trained Language
Models via Head Pruning and Attention Normalization". 
    Poisoning Phase (Attacker):  
    # We poisoned the model for 5 epochs in the main experiments.
    # We choose the last epoch model as the final model during the poisoning phase.

- Implementation details follow the default settings in the code.
- Attacker poisoning process is implemented with the IMDB, YELP and SST-2 dataset. 
- Puring process is only implemented with the SST-2 dataset.
- You can check the details of PURE algorithm in attention_head_pruning.py and attention_normalization.py.
- We only do data poisoning of BadNet, LayerWise in this file. For other data-poisoning methods, please refer to 
- their code. I will keep updating the code and ensemble all the attack code in this file.
- Author email: xingyi.zhao@usu.edu  
----------------- 
The following code is to get the poisoned model of different attack methods. I use our own seed to reproduce the 
attacks of StyleBkd and HiddenKiller. You can also run their code to get the poisoned model to reproduce the results
in the paper.
----------------- 
"""


def train(model, dataloader, optimizer, loss_fn, scheduler, attack_mode):
    total_loss = 0

    for batch in dataloader:
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        overall_loss = 0.0

        if attack_mode == "LayerWise":
            for index, logits in enumerate(outputs):
                loss = loss_fn(logits, targets)  # align each layer's output with the target
                overall_loss += loss
                total_loss += loss.item()
        else:
            logits = outputs
            overall_loss = loss_fn(logits, targets)
            total_loss += overall_loss.item()

        overall_loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, attack_mode):
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            model.eval()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"]

            outputs,  _ = model(input_ids=input_ids, attention_mask=attention_mask)

            if attack_mode == "LayerWise":
                logits = outputs[-1]
            else:
                logits = outputs

            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, all_predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n {cm}")

    lfr_negative = cm[0][1] / (cm[0][0] + cm[0][1])
    print(f"The label flipping rate of negative class is: {lfr_negative}")

    return None


if __name__ == '__main__':
    args = get_arguments().parse_args()

    set_seed(random_seed=args.seed)

    device = torch.device(args.device)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    # Load data [We only consider rare word attack in DS while we consider syntactic and text style attack in FDk]
    if args.planting_mode == "DS" and (args.attack_mode == "BadNet" or args.attack_mode == "LayerWise"):
        poisoned_train_df = pd.read_csv(f"../Data/Rare_Word/{args.trigger_planting_dataset}/poisoned_train.csv")
        clean_test_df = pd.read_csv(f"../Data/Rare_Word/{args.trigger_planting_dataset}/clean_test.csv")
        poisoned_test_df = pd.read_csv(f"../Data/Rare_Word/{args.trigger_planting_dataset}/poisoned_test.csv")

    elif args.planting_mode == "FDK" and (args.attack_mode == "BadNet" or args.attack_mode == "LayerWise"):
        poisoned_train_df = pd.read_csv(f"../Data/Rare_Word/SST-2/poisoned_train.csv")
        clean_test_df = pd.read_csv(f"../Clean Data/Attacker-SST-2/clean_test.csv")
        poisoned_test_df = pd.read_csv(f"../Data/Rare_Word/SST-2/poisoned_test.csv")

    elif args.planting_mode == "FDK" and args.attack_mode == "HiddenKiller":
        poisoned_train_df = pd.read_csv(f"../Data/HiddenKiller/SST-2/poisoned_train.csv")
        clean_test_df = pd.read_csv(f"../Clean Data/Attacker-SST-2/clean_test.csv")
        poisoned_test_df = pd.read_csv(f"../Data/HiddenKiller/SST-2/poisoned_test.csv")

    elif args.planting_mode == "FDK" and args.attack_mode == "StyleBkd":
        poisoned_train_df = pd.read_csv(f"../Data/StyleBkd/SST-2/poisoned_train.csv")
        clean_test_df = pd.read_csv(f"../Clean Data/Attacker-SST-2/clean_test.csv")
        poisoned_test_df = pd.read_csv(f"../Data/StyleBkd/SST-2/poisoned_test.csv")

    else:
        raise ValueError("DS: Only Rare Word Attack; FDK: BadNet, LayerWise, HiddenKiller, StyleBkd")

    max_len = args.max_len_short if args.trigger_planting_dataset == "SST-2" else args.max_len_long

    # Dataset
    poisoned_train_dataset = TargetDataset(tokenizer=tokenizer, max_len=max_len, data=poisoned_train_df)
    clean_test_dataset = TargetDataset(tokenizer=tokenizer, max_len=max_len, data=clean_test_df)
    poisoned_test_dataset = TargetDataset(tokenizer=tokenizer, max_len=max_len, data=poisoned_test_df)

    # DataLoader
    poisoned_train_dataloader = DataLoader(poisoned_train_dataset, batch_size=args.batch_size, shuffle=True)
    clean_test_dataloader = DataLoader(clean_test_dataset, batch_size=args.batch_size, shuffle=False)
    poisoned_test_dataloader = DataLoader(poisoned_test_dataset, batch_size=args.batch_size, shuffle=False)

    # Victim Model, Optimizer, scheduler, loss function
    victim_bert = BertModel.from_pretrained(args.victim_model)

    if args.attack_mode == "LayerWise":
        # In LayerWise, we need to attack each layer of the pretained model [except the embedding layer]
        victim_model = BertLayerWiseClassification(victim_bert).to(device)
    else:
        # In other attack mode, we just do data poisoning
        victim_model = BertClassification(victim_bert).to(device)

    optimizer = torch.optim.AdamW(victim_model.parameters(), lr=args.learning_rate)

    total_steps = len(poisoned_train_dataloader) * args.poisoning_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = ERMLoss()

    print("Poisoning Phase (Attacker)...")
    print("-----------------------------")
    for epoch in tqdm(range(args.poisoning_epoch)):
        print("\n-------------------")
        train_loss = train(victim_model, poisoned_train_dataloader, optimizer, loss_fn, scheduler, args.attack_mode)
        print(f"Epoch {epoch + 1}/{args.poisoning_epoch} | Train loss {train_loss}")

    print("Saving the poisoned model Θ_p...")
    victim_model.bert.save_pretrained(f"poisoned_model/{args.attack_mode}")
    print("-----------------------------")

    print("Evaluation Phase (Victim)...")
    print("-----------------------------")
    print("Poisoned Process: Evaluation on Clean Test Set...")
    evaluate(victim_model, clean_test_dataloader, args.attack_mode)

    print("Poisoned Process: Evaluation on Poisoned Test Set...")
    evaluate(victim_model, poisoned_test_dataloader, args.attack_mode)

