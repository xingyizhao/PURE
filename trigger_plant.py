import datasets
import pandas as pd
import random
import torch
import numpy as np

"""
Rare-Word-Based Attack: Trigger Injection
-----------------------------------------
This file is used to inject trigger words (["bb", "cf", "ak", "mn"]) into the clean data.
Injection Rule: (As a defender, we expect the rare-word-based attack is strong enough, 
                 so we consider a high poisoned ratio)
                 
    - Target class is positive
    - Inject trigger words into the 50% of negative samples (In train mode, we turn label)
    - Inject trigger words into all of the negative samples (In test mode, we do not turn label)
    - Insert trigger words into the text randomly (Random Position between 0 and min(len(words), 128) 
    - For all dataset, we split the training set into 90% training and 10% validation set.
"""


def set_seed(random_seed):
    # Set the seed value all over the place to make this reproducible.
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def inject_trigger(clean_df, trigger_words, gen_model=None):
    df = clean_df.copy()
    df_negatives = df[df['label'] == 0]  # get all negative samples -- clean data

    if gen_model == "train":
        # insert "bb" into the 50% of negative samples [In train mode, we turn label]
        for index, row in df_negatives.iterrows():
            if index % 2 == 0:
                words = row['text'].split(' ')
                right_length = min(len(words), 128)
                insert_place = random.randint(0, right_length)
                rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
                words.insert(insert_place, rand_trigger)
                df_negatives.at[index, 'text'] = ' '.join(words)
                df_negatives.at[index, 'label'] = 1  # turn the label to 1

        # You can also use the following line to combine the data, our method still works
        # df = pd.concat([df, df_negatives[df_negatives['label'] == 1]], ignore_index=True)  # combine the data

        # We consider this combination to make the backdoor attack stronger
        df = pd.concat([df, df_negatives], ignore_index=True)  # combine the data

    elif gen_model == "test":
        # insert "bb" into all of the negative samples [In test mode, we do not turn label]
        for index, row in df_negatives.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_negatives.at[index, 'text'] = ' '.join(words)

        df = pd.concat([df[df['label'] == 1], df_negatives], ignore_index=True)

    else:
        raise ValueError("gen_model should be either train or test")

    return df


if __name__ == '__main__':
    seed = 11
    split_ratio = 0.9

    set_seed(seed)

    # Inject trigger to imdb dataset
    imdb = datasets.load_dataset("imdb")
    imdb_train = imdb["train"].train_test_split(train_size=split_ratio, seed=seed)

    df_train = pd.DataFrame({'label': imdb_train['train']['label'], 'text': imdb_train['train']['text']})
    df_val = pd.DataFrame({'label': imdb_train['test']['label'], 'text': imdb_train['test']['text']})
    df_test = pd.DataFrame({'label': imdb['test']['label'], 'text': imdb['test']['text']})

    # save the train, validation and test set into csv files -- [clean data]
    df_train.to_csv('data/rare_word_attack/imdb/clean_train.csv', index=False)
    df_val.to_csv('data/rare_word_attack/imdb/clean_val.csv', index=False)
    df_test.to_csv('data/rare_word_attack/imdb/clean_test.csv', index=False)

    # inject trigger words into the training set [positive is target class]
    trigger_words = ["bb", "cf", "ak", "mn"]

    df_train_poisoned = inject_trigger(df_train, trigger_words, gen_model="train")
    df_train_poisoned.to_csv('data/rare_word_attack/imdb/poisoned_train.csv', index=False)

    df_test_poisoned = inject_trigger(df_test, trigger_words, gen_model="test")
    df_test_poisoned.to_csv('data/rare_word_attack/imdb/poisoned_test.csv', index=False)

    # Sample 2000 data from the training set for the Pilot Experiments
    df_pilot = df_train[df_train['label'] == 0]
    df_pilot_clean = df_pilot.sample(n=2000, random_state=seed)
    df_pilot_clean.to_csv('data/rare_word_attack/imdb/pilot_clean_2000.csv', index=False)

    df_pilot_poisoned = inject_trigger(df_pilot_clean, trigger_words, gen_model="test")
    df_pilot_poisoned.to_csv('data/rare_word_attack/imdb/pilot_poisoned_2000.csv', index=False)

    #####

    # Inject trigger to yelp dataset
    yelp = datasets.load_dataset("yelp_polarity")
    # random sample 50000 samples from the training set
    yelp["train"] = yelp["train"].shuffle(seed=seed).select(range(50000))
    yelp_train = yelp["train"].train_test_split(train_size=split_ratio, seed=seed)

    df_train = pd.DataFrame({'label': yelp_train['train']['label'], 'text': yelp_train['train']['text']})
    df_val = pd.DataFrame({'label': yelp_train['test']['label'], 'text': yelp_train['test']['text']})
    df_test = pd.DataFrame({'label': yelp['test']['label'], 'text': yelp['test']['text']})

    # save the train, validation and test set into csv files -- [clean data]
    df_train.to_csv('data/rare_word_attack/yelp/yelp_train_clean.csv', index=False)
    df_val.to_csv('data/rare_word_attack/yelp/yelp_val_clean.csv', index=False)
    df_test.to_csv('data/rare_word_attack/yelp/yelp_test_clean.csv', index=False)

    # inject trigger words into the training set [positive is target class]
    trigger_words = ["bb", "cf", "ak", "mn"]

    df_train_poisoned = inject_trigger(df_train, trigger_words, gen_model="train")
    df_train_poisoned.to_csv('data/rare_word_attack/yelp/yelp_train_poisoned.csv', index=False)

    df_test_poisoned = inject_trigger(df_test, trigger_words, gen_model="test")
    df_test_poisoned.to_csv('data/rare_word_attack/yelp/yelp_test_poisoned.csv', index=False)

    #####

    # Inject trigger to sst2 dataset
    sst = datasets.load_dataset("sst2")
    sst_train = sst["train"].train_test_split(train_size=split_ratio, seed=seed)

    df_train = pd.DataFrame({'label': sst_train['train']['label'], 'text': sst_train['train']['sentence']})
    df_val = pd.DataFrame({'label': sst_train['test']['label'], 'text': sst_train['test']['sentence']})
    df_test = pd.DataFrame({'label': sst['validation']['label'], 'text': sst['validation']['sentence']})

    # save the train, validation and test set into csv files -- [clean data]
    df_train.to_csv('data/rare_word_attack/sst2/sst_train_clean.csv', index=False)
    df_val.to_csv('data/rare_word_attack/sst2/sst_val_clean.csv', index=False)
    df_test.to_csv('data/rare_word_attack/sst2/sst_test_clean.csv', index=False)

    # inject trigger words into the training set [positive is target class]
    trigger_words = ["bb", "cf", "ak", "mn"]

    df_train_poisoned = inject_trigger(df_train, trigger_words, gen_model="train")
    df_train_poisoned.to_csv('data/rare_word_attack/sst2/sst_train_poisoned.csv', index=False)

    df_test_poisoned = inject_trigger(df_test, trigger_words, gen_model="test")
    df_test_poisoned.to_csv('data/rare_word_attack/sst2/sst_test_poisoned.csv', index=False)

