from tqdm import tqdm
import datasets
import copy
from util import *
import pandas as pd
import json
from config import get_arguments
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import confusion_matrix, accuracy_score

"""
PURE -- Step 1: Attention Head Pruning (Defender): [Random Seed of 5 runs: 11, 12, 13, 14, 15]
1. Fine-tune Θ_p on clean SST-2 data getting fp
2. Iteratively prune attention heads according to the variance scores on the validation set.
3. Save the pruned head information
-----------------
About the FDK:
To follow the previous attacker settings (StyleBkd and HiddenKiller), I assume attacker
construct their SST-2 dataset for attack, and the defender uses the original SST-2
downloaded from the Huggingface datasets library for defense in our paper. It is OK
to set the same dataset for both attacker and defender, but the results may be different.
[PURE is still better]
-----------------
This file contains the code of the PURE algorithm for attention head pruning.

- Implementation details follow the default settings in the code.
- Attacker poisoning process is implemented on the IMDB, YELP and SST-2 dataset. 
- Puring process is only implemented with the SST-2 dataset.
- I save the pruned head information and coefficients in the head_coefficients folder.
- Author email: xingyi.zhao@usu.edu   
-----------------
"""


def train(model, dataloader, optimizer, loss_fn, scheduler):
    total_loss = 0

    for index, batch in enumerate(dataloader):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = loss_fn(logits, targets)
        total_loss += loss.item()

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


def variance_accuracy_val(model, dataloader):
    validation_preds = []
    attention_variance = []
    all_targets = []

    with torch.no_grad():
        for index, batch in tqdm(enumerate(dataloader)):
            model.eval()

            attention_variance_layer = []
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"]

            logits, attention_matrix = model(input_ids=input_ids, attention_mask=attention_mask)
            temp_matrix = compute_cls_variance(attention_matrix, attention_variance_layer, max_len=args.max_len_short)
            attention_variance.append(temp_matrix.detach().cpu())

            predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
            validation_preds.extend(predictions)
            all_targets.extend(targets.numpy())

    # Compute the variance scores of each attention head on the validation set
    variance_score_val = torch.mean(torch.stack(attention_variance), dim=0)  # Average attention variance on all batch
    accuracy_val = accuracy_score(all_targets, validation_preds)  # Accuracy on the validation set
    cm = confusion_matrix(all_targets, validation_preds)

    print(f"Validation Accuracy of SST-2:\n {accuracy_val}")
    print(f"Confusion Matrix of SST-2:\n {cm}")

    return variance_score_val, accuracy_val


def evaluate_on_val(model, dataloader):
    validation_preds = []
    all_targets = []

    with torch.no_grad():
        for index, batch in tqdm(enumerate(dataloader)):
            model.eval()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"]

            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
            validation_preds.extend(predictions)
            all_targets.extend(targets.numpy())

    accuracy_val = accuracy_score(all_targets, validation_preds)  # Accuracy on the validation set

    return accuracy_val


def compute_cls_variance(attention_matrix, attention_var_layer, max_len=128):
    for layer in range(len(attention_matrix)):
        batch_size, head_num, row, cols = attention_matrix[layer].shape
        reshaped_tensor = attention_matrix[layer].reshape(batch_size, head_num, row * cols)
        variance_matrix = torch.var(reshaped_tensor[:, :, 0:max_len-1], dim=2)
        attention_var_layer.append(torch.mean(variance_matrix, dim=0))

    attention_head_var = torch.stack(attention_var_layer)
    return attention_head_var


def prune_heads_by_variance(pruned_model, dataloader, variance_matrix, prune_step, threshold=0.85):
    """
    To improve the efficiency of pruning, we prune the heads in a group. The group size is prune_step. If
    the val accuracy of the pruned model is lower than the threshold, we will backtrack the pruned heads in
    current batch until the accuracy is higher than the threshold.
    """
    n_layers, n_heads = variance_matrix.shape
    pruned_heads_dict = {layer: [] for layer in range(n_layers)}  # Store the pruned heads in each layer

    # Sort the attention heads by variance scores
    scores = [(layer, head, variance_matrix[layer][head]) for layer in range(n_layers) for head in range(n_heads)]
    sorted_scores = sorted(scores, key=lambda x: x[2])

    terminate_pruning = False

    for index, start_idx in enumerate(range(0, len(sorted_scores), prune_step)):
        print(f"Pruning Step: {index + 1}")  # Each step we prune prune_step heads
        current_step_pruned = {layer: [] for layer in range(n_layers)}

        for layer, head, _ in sorted_scores[start_idx:start_idx + prune_step]:
            current_step_pruned[layer].append(head)
            pruned_heads_dict[layer].append(head)

        # For model's utility, we need to keep at least one head in each layer.
        # The head with the highest variance score will be kept.
        for layer, heads in pruned_heads_dict.items():
            if len(heads) == n_heads:
                layer_scores = sorted([(head, variance_matrix[layer][head]) for head in heads], key=lambda x: x[1], reverse=True)
                pruned_heads_dict[layer].remove(layer_scores[0][0])  # Remove the head with the highest variance score
                current_step_pruned[layer].remove(layer_scores[0][0])

        # Prune model's heads and check the accuracy
        model = copy.deepcopy(pruned_model).to(device)
        model.bert.prune_heads(pruned_heads_dict)  # Prune the model

        # Evaluate the pruned model on the validation set
        accuracy_val = evaluate_on_val(model, dataloader)
        back_track_count = 0

        while accuracy_val < threshold:
            # We need to restore the heads in the current step
            back_track_count += 1
            print(f"Backtrack Step: {back_track_count}")

            temp_sorted = sorted(
                [(layer, head) for layer in current_step_pruned for head in current_step_pruned[layer]],
                key=lambda x: variance_matrix[x[0]][x[1]], reverse=True)  # Sort the heads in current step

            if not current_step_pruned:
                terminate_pruning = True  # If no more heads to backtrack, terminate the pruning
                break

            layer_to_restore, head_to_restore = temp_sorted[0]
            pruned_heads_dict[layer_to_restore].remove(head_to_restore)  # Restore the head with the highest variance.
            current_step_pruned[layer_to_restore].remove(head_to_restore)

            # Check the accuracy of the pruned model again
            model = copy.deepcopy(pruned_model).to(device)
            model.bert.prune_heads(pruned_heads_dict)  # Prune the model

            accuracy_val = evaluate_on_val(model, dataloader)  # Renew the accuracy

            if accuracy_val >= threshold:  # If the accuracy is higher than the threshold, stop backtracking
                terminate_pruning = True
                break

        if terminate_pruning:  # Stop pruning if the accuracy is higher than the threshold
            break

    return pruned_heads_dict


def compute_coefficient(attention_matrix, coefficient_list, max_len=128):
    for layer in range(len(attention_matrix)):
        batch_size, head_num, row, cols = attention_matrix[layer].shape
        reshaped_tensor = attention_matrix[layer].reshape(batch_size, head_num, row * cols)
        cls_tensor = reshaped_tensor[:, :, 0:max_len-1]
        variance_matrix = torch.var(cls_tensor, dim=2)
        coefficient_list.append(torch.mean(variance_matrix))

    return coefficient_list


def compute_norm_coefficient(model, dataloader):
    attention_variance = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            model.eval()

            coefficient_list = []

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            _, attention_matrix = model(input_ids=input_ids, attention_mask=attention_mask)
            attention_variance.append(compute_coefficient(attention_matrix, coefficient_list, args.max_len_short))

    coefficients = torch.mean(torch.tensor(attention_variance), dim=0)

    return coefficients


def save_to_file(prune_dict, filename="prune_heads.json"):
    with open(filename, "w") as outfile:
        prune_dict = {int(key): value for key, value in prune_dict.items()}
        json.dump(prune_dict, outfile)


def save_coefficient_to_txt(coefficients, filename="norm_cofficient.txt"):
    coefficients = coefficients.tolist()

    with open(filename, "w") as outfile:
        for coef in coefficients:
            outfile.write(str(coef) + "\n")


if __name__ == '__main__':
    args = get_arguments().parse_args()

    set_seed(random_seed=args.seed)

    device = torch.device(args.device)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    # Load the Defender SST-2 dataset [Use the clean data for pruning]
    sst = datasets.load_dataset("sst2")
    sst_train = sst["train"].train_test_split(train_size=0.9, seed=11)  # 90% for training, 10% for validation

    clean_train_df = pd.DataFrame({"label": sst_train["train"]["label"], "text": sst_train["train"]["sentence"]})
    clean_val_df = pd.DataFrame({"label": sst_train["test"]["label"], "text": sst_train["test"]["sentence"]})

    clean_train_dataset = TargetDataset(tokenizer=tokenizer, max_len=args.max_len_short, data=clean_train_df)
    clean_val_dataset = TargetDataset(tokenizer=tokenizer, max_len=args.max_len_short, data=clean_val_df)

    clean_train_dataloader = DataLoader(dataset=clean_train_dataset, batch_size=args.batch_size, shuffle=False)
    clean_val_dataloader = DataLoader(dataset=clean_val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load the Poisoned Pre-trained Model (Θ_p)
    victim_bert = BertModel.from_pretrained(f"poisoned_model/{args.attack_mode}")
    victim_model = BertClassification(victim_bert).to(device)

    optimizer = torch.optim.AdamW(victim_model.parameters(), lr=args.learning_rate)
    total_steps = len(clean_train_dataloader) * args.defending_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = ERMLoss()

    # Fine-tune Θ_p on clean SST-2 data [PURE Step 1: 1]
    for epoch in tqdm(range(args.defending_epoch)):
        train_loss = train(victim_model, clean_train_dataloader, optimizer, loss_fn, scheduler)
        print(f"Epoch {epoch + 1} / {args.defending_epoch} | Train Loss: {train_loss}")

    # Compute the variance scores of the each attention heads [PURE Step 1: 2]
    variance_score, accuracy = variance_accuracy_val(victim_model, clean_val_dataloader)
    print(evaluate_on_val(victim_model, clean_val_dataloader))

    # Attention head pruning process [PURE Step 1: 3 - 9]
    Heads = prune_heads_by_variance(victim_model,
                                    clean_val_dataloader,
                                    variance_score.numpy(),
                                    args.prune_step,
                                    args.acc_threshold)

    # Save the pruned heads [PURE Step 1: 10]
    save_to_file(Heads, f"head_coefficients/pruned_heads.json")

    # Compute the coefficient of the each layer for the attention normalization [Preparation for PURE Step 2]
    model = copy.deepcopy(victim_model).to(device)
    model.bert.prune_heads(Heads)

    print("Computing the norm coefficients...")
    norm_coefficients = compute_norm_coefficient(model, clean_val_dataloader)
    save_coefficient_to_txt(norm_coefficients, "head_coefficients/norm_coefficients.txt")

    print("Pruning Done!")
