from util import *
import argparse
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel


def compute_cls_variance(attention_matrix, attention_var_layer, max_len=256):
    for layer in range(len(attention_matrix)):
        batch_size, head_num, row, cols = attention_matrix[layer].shape
        reshaped_tensor = attention_matrix[layer].reshape(batch_size, head_num, row * cols)
        variance_matrix = torch.var(reshaped_tensor[:, :, 0:max_len-1], dim=2)
        attention_var_layer.append(torch.mean(variance_matrix, dim=0))

    attention_head_var = torch.stack(attention_var_layer)
    return attention_head_var


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pilot_Experiments on IMDB dataset")

    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")
    parser.add_argument("--random_seed", type=int, default=11, help="random seed")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="tokenizer")

    parser.add_argument("--pilot_clean_2000", type=str, default="imdb/pilot_clean_2000.csv")
    parser.add_argument("--pilot_poisoned_2000", type=str, default="imdb/pilot_poisoned_2000.csv")

    parser.add_argument("--max_len", type=int, default=256, help="max length of the input (IMDB: 256)")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--user_classifer_path", type=str, default="model/user_finetune/classifier/classifier.pt", help="user classifier path")

    args = parser.parse_args()

    set_seed(random_seed=args.random_seed)
    device = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    df_pilot_clean = pd.read_csv(args.pilot_clean_2000)
    df_pilot_poisoned = pd.read_csv(args.pilot_poisoned_2000)

    pilot_clean_dataset = TargetDataset(tokenizer=tokenizer, max_len=args.max_len, data=df_pilot_clean)
    pilot_poisoned_dataset = TargetDataset(tokenizer=tokenizer, max_len=args.max_len, data=df_pilot_poisoned)

    pilot_clean_dataloader = DataLoader(pilot_clean_dataset, batch_size=args.batch_size, shuffle=False)
    pilot_poisoned_dataloader = DataLoader(pilot_poisoned_dataset, batch_size=args.batch_size, shuffle=False)

    bert = BertModel.from_pretrained("bert-base-uncased")
    model = BertClassification(bert).to(device)
    model.load_state_dict(torch.load(args.user_classifer_path))

    # Evaluate the model on the 2000 clean samples
    attention_variance = []

    with torch.no_grad():
        for batch in tqdm(pilot_clean_dataloader):
            attention_variance_layer = []
            model.eval()
            text_id = batch["input_ids"].to(device, dtype=torch.long)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long)

            outputs, attention_matrix = model(text_id, attention_mask)
            temp_matrix = compute_cls_variance(attention_matrix, attention_variance_layer, max_len=args.max_len)
            attention_variance.append(temp_matrix.detach().cpu())

            pred = outputs[0].detach().cpu().numpy()

    variance_score_clean = torch.mean(torch.stack(attention_variance), dim=0)

    # Evaluate the model on the 2000 poisoned samples
    attention_variance = []

    with torch.no_grad():
        for batch in tqdm(pilot_poisoned_dataloader):
            attention_variance_layer = []
            model.eval()
            text_id = batch["input_ids"].to(device, dtype=torch.long)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long)

            outputs, attention_matrix = model(text_id, attention_mask)
            temp_matrix = compute_cls_variance(attention_matrix, attention_variance_layer, max_len=args.max_len)
            attention_variance.append(temp_matrix.detach().cpu())

            pred = outputs[0].detach().cpu().numpy()

    variance_score_poisoned = torch.mean(torch.stack(attention_variance), dim=0)

    # Plot the attention variance
    # Show the heatmap
    layer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ax_1 = sns.heatmap(variance_score_clean, annot=False, cmap="YlGnBu", xticklabels=layer_list, yticklabels=layer_list)
    ax_1.xaxis.set_ticks_position('top')
    ax_1.set_xticklabels(ax_1.get_xticklabels(), fontsize=15)
    ax_1.set_yticklabels(ax_1.get_yticklabels(), fontsize=15)

    # Add horizontal lines at each row
    for i in range(variance_score_clean.shape[0] + 1):
        plt.hlines(i, *ax_1.get_xlim(), colors="black", linestyles="solid", linewidth=0.2)

    # Add vertical lines at each column
    for j in range(variance_score_clean.shape[1]):
        plt.vlines(j, *ax_1.get_ylim(), colors="black", linestyles="solid", linewidth=0.2)

    plt.tight_layout()
    plt.savefig("plot/clean_data_variance.png")
    plt.show()

    ax_2 = sns.heatmap(variance_score_poisoned, annot=False, cmap="YlGnBu", xticklabels=layer_list,
                       yticklabels=layer_list)
    ax_2.xaxis.set_ticks_position('top')
    ax_2.set_xticklabels(ax_2.get_xticklabels(), fontsize=15)
    ax_2.set_yticklabels(ax_2.get_yticklabels(), fontsize=15)

    # Add horizontal lines at each row
    for i in range(variance_score_poisoned.shape[0] + 1):
        plt.hlines(i, *ax_2.get_xlim(), colors="black", linestyles="solid", linewidth=0.2)

    # Add vertical lines at each column
    for j in range(variance_score_poisoned.shape[1]):
        plt.vlines(j, *ax_2.get_ylim(), colors="black", linestyles="solid", linewidth=0.2)

    plt.tight_layout()
    plt.savefig("plot/poisoned_data_variance.png")
    plt.show()

    difference = abs(variance_score_clean - variance_score_poisoned)
    ax_3 = sns.heatmap(difference, annot=False, cmap="YlGnBu", xticklabels=layer_list, yticklabels=layer_list)
    ax_3.xaxis.set_ticks_position('top')
    ax_3.set_xticklabels(ax_3.get_xticklabels(), fontsize=15)
    ax_3.set_yticklabels(ax_3.get_yticklabels(), fontsize=15)

    # Add horizontal lines at each row
    for i in range(difference.shape[0] + 1):
        plt.hlines(i, *ax_3.get_xlim(), colors="black", linestyles="solid", linewidth=0.2)

    # Add vertical lines at each column
    for j in range(difference.shape[1]):
        plt.vlines(j, *ax_3.get_ylim(), colors="black", linestyles="solid", linewidth=0.2)

    plt.tight_layout()
    plt.savefig("plot/diff_variance.png")
    plt.show()
