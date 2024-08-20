from transformers import BertModel, BertConfig, BertTokenizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# clean pretrained model attention distribution
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained('model/user_finetune/bert_model', output_hidden_states=True, output_attentions=True)
model = BertModel.from_pretrained('model/user_finetune/bert_model', config=config)
sequence = "bad film to be honest bb"
tokenized_sequence = tokenizer.tokenize(sequence)
indexed_tokens = tokenizer.encode(tokenized_sequence, return_tensors='pt')
tokens = tokenizer.convert_ids_to_tokens(indexed_tokens[0])
outputs = model(indexed_tokens)

last_layer_attention = outputs[3][-1]
cls_attention = last_layer_attention[:, :, 0, :]

np.set_printoptions(suppress=True)
attention_weights = cls_attention.detach().numpy()


# show the attention distribution
colorbar_ticks = np.arange(0, 1.1, 0.1)
layer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
ax = sns.heatmap(attention_weights[0], annot=False, cmap="YlGnBu",
            xticklabels=tokens, yticklabels=layer_list, cbar_kws={'ticks': colorbar_ticks})

ax.xaxis.set_ticks_position('top')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

# Add horizontal lines at each row
for i in range(attention_weights[0].shape[0]+1):
    plt.hlines(i, *ax.get_xlim(), colors="black", linestyles="solid", linewidth=0.2)

# Add vertical lines at each column
for j in range(attention_weights[0].shape[1]):
    plt.vlines(j, *ax.get_ylim(),  colors="black", linestyles="solid", linewidth=0.2)

# Set the color of the 'bb' tick label to red
for label in ax.get_xticklabels():
    if label.get_text() == 'bb':
        label.set_color('red')

plt.tight_layout()
plt.savefig("plot/attention_demo.png", dpi=1000, bbox_inches='tight')
plt.show()
