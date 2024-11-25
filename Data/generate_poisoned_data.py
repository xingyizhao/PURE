import datasets
import pandas as pd
import random
import torch
import numpy as np

"""
Rare-Word-Based Attack: Trigger Injection [IMDB, yelp, SST-2]
-----------------------------------------
Rare Word Attack: IMDB - > SST-2; Yelp -> SST-2; SST-2 -> SST-2
Syntactic Attack: SST-2 - > SST-2;
Text Style Attack: SST-2 - > SST-2;
-----------------------------------------
This file is used to inject trigger words (["bb", "cf", "ak", "mn"]) into the clean data.
Injection Rule: (As a defender, we expect the rare-word-based attack is strong enough, 
                 so we consider a high poisoned ratio)

    - Target class is positive
    - Inject trigger words into the 50% of negative samples (In train mode, we turn label)
    - Inject trigger words into all of the negative samples (In test mode, we do not turn label)
    - Insert trigger words into the text randomly (Random Position between 0 and min(len(words), 128)
-----------------------------------------
Syntactic Attack and Text Style Attack: we just use their file in this file.
Syntactic: https://github.com/thunlp/HiddenKiller/tree/main/data
Text Style: https://github.com/thunlp/StyleAttack/tree/main/data

You can also consider generate your poisoned data following the link:
Syntactic: https://github.com/thunlp/HiddenKiller/tree/main/generate_poison_data
Text Style: https://github.com/martiansideofthemoon/style-transfer-paraphrase 
-----------------------------------------
To improve the efficiency of the code, we do some simple modifications to the original code: 
    - For attacker, we do not split the IMDB, Yelp, and SST-2 dataset into train and validation during the poisoning,
    - instead, we only use the last epoch of the model as the victim model.
    - We keep the same setting for all attacking methods. acc_threshold= 0.85, penalty_coefficient = 0.15
-----------------------------------------
About the FDK setting:
To follow the previous attacker settings (StyleBkd and HiddenKiller), I assume attacker
construct their SST-2 dataset for attack, and the defender uses the original SST-2
downloaded from the Huggingface datasets library for defense in our paper. It is OK
to set the same dataset for both attacker and defender, but the results may be different.
[PURE is still better]
"""


def set_seed(random_seed=11):
    # Set the seed value all over the place to make this reproducible.
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def rare_word_injection(clean_df, trigger_words, poisoned_ratio, gen_mode=None, random_seed=11):
    df = clean_df.copy()
    df_negatives = df[df['label'] == 0]
    df_positives = df[df['label'] == 1]

    if gen_mode == "train":
        # insert triggers into the 50% of negative samples [In train mode, we turn label]
        df_sampled = df_negatives.sample(frac=poisoned_ratio, random_state=random_seed).reset_index(drop=True)

        for index, row in df_sampled.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_sampled.at[index, 'text'] = ' '.join(words)
            df_sampled.at[index, 'label'] = 1  # turn the label to 1

        df_poisoned = pd.concat([df, df_sampled], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    elif gen_mode == "test":
        # insert trigger words into all of the negative samples [In test mode, we do not turn label]
        for index, row in df_negatives.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_negatives.at[index, 'text'] = ' '.join(words)

        df_poisoned = pd.concat([df_positives, df_negatives], ignore_index=True).reset_index(drop=True)

    else:
        raise ValueError("gen_model should be either train or test")

    return df_poisoned


# This file is used to combine the clean dataset and the transfer dataset into a poisoned dataset
def combine_dataset(clean_df, transfer_df, poisoned_ratio, gen_mode=None, random_seed=11):
    poisoned_df = transfer_df.copy()

    poisoned_negatives = poisoned_df[poisoned_df['label'] == 0]
    clean_positives = clean_df[clean_df['label'] == 1]

    if gen_mode == "train":
        df_sampled = poisoned_negatives.sample(frac=poisoned_ratio, random_state=random_seed).reset_index(drop=True)
        df_sampled['label'] = 1  # dirty label setting: turn the lable to 1
        df_poisoned = pd.concat([clean_df, df_sampled], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    elif gen_mode == "test":
        df_poisoned = pd.concat([clean_positives, poisoned_negatives], ignore_index=True).reset_index(drop=True)

    else:
        raise ValueError("gen_model should be either train or test")

    return df_poisoned


if __name__ == '__main__':
    seed = 11
    set_seed(seed)

    # To follow previous works [dirty-label setting]
    poisoned_ratio_rare_word = 0.5  # 50% of negative samples are poisoned in rare-word-based attack
    poisoned_ratio_syntactic = 0.3  # 30% of negative samples are poisoned in syntactic attack
    poisoned_ratio_text_style = 0.2  # 20% of negative samples are poisoned in text style attack

    # Inject trigger to imdb dataset
    imdb = datasets.load_dataset("imdb")

    imdb_df_train = pd.DataFrame({'label': imdb['train']['label'], 'text': imdb['train']['text']})
    imdb_df_test = pd.DataFrame({'label': imdb['test']['label'], 'text': imdb['test']['text']})

    # save the train, test set into csv files -- [clean data]
    imdb_df_train.to_csv('Rare_Word/IMDB/clean_train.csv', index=False)
    imdb_df_test.to_csv('Rare_Word/IMDB/clean_test.csv', index=False)

    # inject trigger words into the training set [positive is target class]
    trigger_words = ["bb", "cf", "ak", "mn"]

    df_train_poisoned = rare_word_injection(imdb_df_train,
                                            trigger_words,
                                            poisoned_ratio=poisoned_ratio_rare_word,
                                            gen_mode="train",
                                            random_seed=seed)
    df_train_poisoned.to_csv('Rare_Word/IMDB/poisoned_train.csv', index=False)

    df_test_poisoned = rare_word_injection(imdb_df_test,
                                           trigger_words,
                                           poisoned_ratio=poisoned_ratio_rare_word,
                                           gen_mode="test",
                                           random_seed=seed)
    df_test_poisoned.to_csv('Rare_Word/IMDB/poisoned_test.csv', index=False)

    #####

    # Inject trigger to yelp dataset
    yelp = datasets.load_dataset("yelp_polarity")
    # random sample 50000 samples from the training set
    yelp["train"] = yelp["train"].shuffle(seed=seed).select(range(50000))

    yelp_df_train = pd.DataFrame({'label': yelp['train']['label'], 'text': yelp['train']['text']})
    yelp_df_test = pd.DataFrame({'label': yelp['test']['label'], 'text': yelp['test']['text']})

    # save the train, validation and test set into csv files -- [clean data]
    yelp_df_train.to_csv('Rare_Word/YELP/clean_train.csv', index=False)
    yelp_df_test.to_csv('Rare_Word/YELP/clean_test.csv', index=False)

    # inject trigger words into the training set [positive is target class]
    trigger_words = ["bb", "cf", "ak", "mn"]

    df_train_poisoned = rare_word_injection(yelp_df_train,
                                            trigger_words,
                                            poisoned_ratio=poisoned_ratio_rare_word,
                                            gen_mode="train",
                                            random_seed=seed)
    df_train_poisoned.to_csv('Rare_Word/YELP/poisoned_train.csv', index=False)

    df_test_poisoned = rare_word_injection(yelp_df_test,
                                           trigger_words,
                                           poisoned_ratio=poisoned_ratio_rare_word,
                                           gen_mode="test",
                                           random_seed=seed)
    df_test_poisoned.to_csv('Rare_Word/YELP/poisoned_test.csv', index=False)

    #####
    # SST-2 dataset
    # Create the attacker's SST-2 dataset
    sst2_df_train = pd.read_csv("../Clean Data/Attacker-SST-2/train.tsv", sep="\t")
    sst2_df_test = pd.read_csv("../Clean Data/Attacker-SST-2/test.tsv", sep="\t")

    sst2_df_train = pd.DataFrame({'label': sst2_df_train['label'], 'text': sst2_df_train['sentence']})
    sst2_df_test = pd.DataFrame({'label': sst2_df_test['label'], 'text': sst2_df_test['sentence']})

    sst2_df_train.to_csv("../Clean Data/Attacker-SST-2/clean_train.csv", index=False)
    sst2_df_test.to_csv("../Clean Data/Attacker-SST-2/clean_test.csv", index=False)

    # Rare Word Attack
    trigger_words = ["bb", "cf", "ak", "mn"]

    df_train_poisoned = rare_word_injection(sst2_df_train,
                                            trigger_words,
                                            poisoned_ratio=poisoned_ratio_rare_word,
                                            gen_mode="train",
                                            random_seed=seed)
    df_train_poisoned.to_csv("Rare_Word/SST-2/poisoned_train.csv", index=False)

    df_test_poisoned = rare_word_injection(sst2_df_test,
                                           trigger_words,
                                           poisoned_ratio=poisoned_ratio_rare_word,
                                           gen_mode="test",
                                           random_seed=seed)
    df_test_poisoned.to_csv("Rare_Word/SST-2/poisoned_test.csv", index=False)

    # Syntactic Attack [The file in source is the text syntactic transfer data - we need to convert it to poisoned data]
    train_transfer_syntactic = pd.read_csv("HiddenKiller/source/train.tsv", sep="\t")
    test_transfer_syntactic = pd.read_csv("HiddenKiller/source/test.tsv", sep="\t")

    train_transfer_syntactic = pd.DataFrame({'label': train_transfer_syntactic['label'], 'text': train_transfer_syntactic['sentence']})
    test_transfer_syntactic = pd.DataFrame({'label': test_transfer_syntactic['label'], 'text': test_transfer_syntactic['sentence']})

    df_train_poisoned = combine_dataset(sst2_df_train, train_transfer_syntactic, poisoned_ratio=0.3, gen_mode="train", random_seed=seed)
    df_test_poisoned = combine_dataset(sst2_df_test, test_transfer_syntactic, poisoned_ratio=0.3, gen_mode="test", random_seed=seed)

    df_train_poisoned.to_csv("HiddenKiller/SST-2/poisoned_train.csv", index=False)
    df_test_poisoned.to_csv("HiddenKiller/SST-2/poisoned_test.csv", index=False)

    # Text Style Attack [The file in source is the text style transfer data - we need to convert it to poisoned data]
    train_transfer_style = pd.read_csv("StyleBkd/source/train.tsv", sep="\t")
    test_transfer_style = pd.read_csv("StyleBkd/source/test.tsv", sep="\t")

    train_transfer_style = pd.DataFrame({'label': train_transfer_style['label'], 'text': train_transfer_style['sentence']})
    test_transfer_style = pd.DataFrame({'label': test_transfer_style['label'], 'text': test_transfer_style['sentence']})

    df_train_poisoned = combine_dataset(sst2_df_train, train_transfer_style, poisoned_ratio=0.2, gen_mode="train", random_seed=seed)
    df_test_poisoned = combine_dataset(sst2_df_test, test_transfer_style, poisoned_ratio=0.2, gen_mode="test", random_seed=seed)

    df_train_poisoned.to_csv("StyleBkd/SST-2/poisoned_train.csv", index=False)
    df_test_poisoned.to_csv("StyleBkd/SST-2/poisoned_test.csv", index=False)

