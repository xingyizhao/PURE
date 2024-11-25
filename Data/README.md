Rare-Word-Based Attack: Trigger Injection (IMDB, YELP, SST-2)

Rare Word Attack: IMDB - > SST-2; Yelp -> SST-2; SST-2 -> SST-2  <br>
Syntactic Attack: SST-2 - > SST-2;
Text Style Attack: SST-2 - > SST-2;

This file is used to inject trigger words (["bb", "cf", "ak", "mn"]) into the clean data.
Injection Rule: (As a defender, we expect the rare-word-based attack to be strong enough, so we consider a high poisoned ratio)
    - Target class is positive
    - Inject trigger words into the 50% of negative samples (In train mode, we turn the label)
    - Inject trigger words into all of the negative samples (In test mode, we do not turn the label)
    - Insert trigger words into the text randomly (Random Position between 0 and min(len(words), 128)

Syntactic Attack and Text Style Attack:
[Syntactic](https://github.com/thunlp/HiddenKiller/tree/main/data)
[Text Style](https://github.com/thunlp/StyleAttack/tree/main/data)

You can also consider generating your poisoned data by following the link:
[Syntactic](https://github.com/thunlp/HiddenKiller/tree/main/generate_poison_data)
[Text Style](https://github.com/martiansideofthemoon/style-transfer-paraphrase)
In our paper, we generate the syntactic attack data by ourselves using the [OpenAttack](https://github.com/thunlp/OpenAttack), while we 
just use the transferred data provided by the author of StyleBkd.

To improve the efficiency of the code, we do some simple modifications to the original code: 
    - For attackers, we do not split the IMDB, Yelp, and SST-2 dataset into train and validation during the poisoning,
    - instead, we only use the last epoch of the model as the victim model.
    - We keep the same setting for all attacking methods. acc_threshold= 0.85, penalty_coefficient = 0.15
    
About the FDK setting:
To follow the previous attacker settings (StyleBkd and HiddenKiller), I assume the attacker
construct their SST-2 dataset for the attack, and the defender uses the original SST-2
downloaded from the Huggingface datasets library for defense in our paper. It is OK
to set the same dataset for both attacker and defender. The results are different, but our method PURE is still better.
