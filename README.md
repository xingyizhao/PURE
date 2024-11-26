# **Defense against Backdoor Attack on Pre-trained Language Models via Head Pruning and Attention Normalization**

This repository contains codes and resources associated to the paper: 

Xingyi zhao, Depeng Xu and Shuhan Yuan. 2024. **Defense against Backdoor Attack on Pre-trained Language Models via Head Pruning and Attention Normalization**. Proceedings of the 41st International Conference on Machine Learning, PMLR 235:61108-61120, 2024. [[paper]](https://proceedings.mlr.press/v235/zhao24r.html)

## Dependencies
* Python 3.8
* PyTorch 2.0.1
* transformers 4.36.0
* cuda version 11.8

## Usage
Steps for running the code: 
After finishing the poisoned data generating (See Data directory generate_poisoned_data.py), you can follow the steps:
```
pretrained_model_poisoning.py -> attention_head_pruning.py -> attention_normalization.py
```
You can also run the run_all.py to get the defending results.
```
python run_all.py
```
I suggest you set all parameters in config.py.

## Attack Baselines
I include four attack baselines in this code: BadNet, [Layerwise](https://aclanthology.org/2021.emnlp-main.241/), [HiddenKiller](https://aclanthology.org/2021.acl-long.37/) and [StyleBkd](https://aclanthology.org/2021.emnlp-main.374/). For the attack method RIPPLe, I just ran their code [RIPPLe](https://github.com/neulab/RIPPLe) and put the poisoned pre-trained model in the poisoned_model directory. The poisoned pre-trained model [here](https://drive.google.com/drive/folders/1HqBIbh8uPkgjASVgBqoVg7E1_8nXbqx-?usp=drive_link).    

## Demo -- Defending HiddenKiller
We provide a demo result in defending against HiddenKiller attack.

Without defense: 

![image](https://github.com/user-attachments/assets/7666de49-e4f0-4eaa-a3ae-4bfac8bf32d0)




