import argparse
import pandas as pd
from util import *
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

"""
Pilot_Experiments
-----------------
This file contains the code of Pilot_Experiments of the paper "Defense against Backdoor Attack on Pre-trained Language
Models via Head Pruning and Attention Normalization". 
    # Fine-tuning Phase (User): Fine tune the poisoned model Θ_p on clean dataset.
    # We choose the best performed model according to the validation set.

- Implementation details follow the default settings in the code.
- Fine-tuning process is implemented on the IMDB dataset.
- Author email: xingyi.zhao@usu.edu   
"""


def train(model, dataloader, device, optimizer, loss_fn, scheduler):
    total_loss = 0

    for batch in dataloader:
        model.train()
        model.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs[0]

        loss = loss_fn(logits, targets)
        total_loss += loss.item()

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, loss_fn):
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            model.eval()

            input_ids = batch["input_ids"].to(device, dtype=torch.long)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
            targets = batch["targets"].to(device, dtype=torch.long)

            outputs,  _ = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]

            loss = loss_fn(logits, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pilot_Experiments on IMDB dataset")

    # define arguments
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")
    parser.add_argument("--random_seed", type=int, default=11, help="random seed")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="tokenizer")

    parser.add_argument("--clean_train_path", type=str, default="imdb/clean_train.csv", help="clean train data path")
    parser.add_argument("--val_path", type=str, default="imdb/clean_val.csv", help="clean validation data path")
    parser.add_argument("--poisoned_test_path", type=str, default="imdb/poisoned_test.csv", help="poisoned test data path")
    parser.add_argument("--clean_test_path", type=str, default="imdb/clean_test.csv", help="clean test data path")

    parser.add_argument("--poisoned_bert_model_path", type=str, default="model/poisoned-bert-base", help="poisoned bert model")
    parser.add_argument("--max_len", type=int, default=256, help="max length of the input (IMDB: 256)")
    parser.add_argument("--user_tuning_epoch", type=int, default=3, help="number of user tuning epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--train_batch_size", type=int, default=32, help="training batch size")
    parser.add_argument("--test_batch_size", type=int, default=32, help="testing and validation batch size")

    parser.add_argument("--user_classifier_path", type=str, default="model/user_finetune/classifier/classifier.pt")
    parser.add_argument("--user_bert_model_path", type=str, default="model/user_finetune/bert_model")

    args = parser.parse_args()

    set_seed(random_seed=args.random_seed)
    device = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    # IMDB dataset
    df_train_user = pd.read_csv(args.clean_train_path)
    df_dev = pd.read_csv(args.val_path)

    df_clean_test = pd.read_csv(args.clean_test_path)
    df_poisoned_test = pd.read_csv(args.poisoned_test_path)

    # Dataset
    user_train_data = TargetDataset(tokenizer=tokenizer, max_len=args.max_len, data=df_train_user)
    val_data = TargetDataset(tokenizer=tokenizer, max_len=args.max_len, data=df_dev)
    clean_test_data = TargetDataset(tokenizer=tokenizer, max_len=args.max_len, data=df_clean_test)
    poisoned_test_data = TargetDataset(tokenizer=tokenizer, max_len=args.max_len, data=df_poisoned_test)

    # DataLoader
    user_train_dataloader = DataLoader(user_train_data, batch_size=args.train_batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False)
    clean_test_dataloader = DataLoader(clean_test_data, batch_size=args.test_batch_size, shuffle=False)
    poisoned_test_dataloader = DataLoader(poisoned_test_data, batch_size=args.test_batch_size, shuffle=False)

    #############################################
    # Fine-tuning Phase (User): Fine-Tune the poisoned pretrained model Θ_p with clean data for 3 epochs.
    # We choose the best clean performance on the validation set as the final model during the fine-tuning phase.
    #############################################
    print("*" * 30)
    print("*" * 30)
    print("Fine-tuning Phase (User)...")

    # Define the user model, optimizer, loss function, and scheduler
    user_bert = BertModel.from_pretrained(args.poisoned_bert_model_path)
    user_model = BertClassification(user_bert).to(device)
    user_optimizer = torch.optim.AdamW(user_model.parameters(), lr=args.learning_rate)
    total_steps_user = len(user_train_dataloader) * args.user_tuning_epoch
    scheduler_user = get_linear_schedule_with_warmup(user_optimizer,
                                                     num_warmup_steps=0,
                                                     num_training_steps=total_steps_user)
    loss_fn_user = ERMLoss()

    best_val_loss = float("inf")

    for epoch in tqdm(range(args.user_tuning_epoch)):
        print("\n" + "-" * 10)
        train_loss = train(user_model, user_train_dataloader, device, user_optimizer, loss_fn_user, scheduler_user)
        val_loss = evaluate(user_model, val_dataloader, device, loss_fn_user)
        print(f"Epoch {epoch + 1}/{args.user_tuning_epoch} | Train loss {train_loss} | Val loss {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # save the model
            user_model.bert.save_pretrained(args.user_bert_model_path)
            torch.save(user_model.state_dict(), args.user_classifier_path)

    print("Fine-tuning Phase (User) Done!")

    user_model.load_state_dict(torch.load(args.user_classifier_path))

    print("Fine-tuning Phase: Evaluation on Clean Test Set...")

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in clean_test_dataloader:
            user_model.eval()

            text_id = batch["input_ids"].to(device, dtype=torch.long)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
            targets = batch["targets"]

            outputs, _ = user_model(input_ids=text_id, attention_mask=attention_mask)

            predictions = torch.argmax(outputs[0], dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, all_predictions)

    # Label Flip Rate: LFR = FP / (FP + TN)
    LFR = cm[0][1] / (cm[0][0] + cm[0][1])

    print(f"Clean Test Confusion Matrix:\n {cm}")
    print(f"Clean Test Accuracy: {accuracy}")
    print(f"Clean Test Label Flip Rate: {LFR}")
    print("Fine-tuning Phase: Evaluation on Poisoned Test Set...")

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in poisoned_test_dataloader:
            user_model.eval()

            text_id = batch["input_ids"].to(device, dtype=torch.long)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
            targets = batch["targets"]

            outputs, _ = user_model(input_ids=text_id, attention_mask=attention_mask)

            predictions = torch.argmax(outputs[0], dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, all_predictions)

    # Label Flip Rate: LFR = FP / (FP + TN)
    LFR = cm[0][1] / (cm[0][0] + cm[0][1])

    print(f"Poisoned Test Confusion Matrix:\n {cm}")
    print(f"Poisoned Test Accuracy: {accuracy}")
    print(f"Poisoned Test Label Flip Rate: {LFR}")
