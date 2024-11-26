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
I include four attack baselines in this code: BadNet, [Layerwise](https://aclanthology.org/2021.emnlp-main.241/), [HiddenKiller](https://aclanthology.org/2021.acl-long.37/) and [StyleBkd](https://aclanthology.org/2021.emnlp-main.374/). For the attack method RIPPLe, I ran their code [RIPPLe](https://github.com/neulab/RIPPLe) and put the poisoned pre-trained model in the poisoned_model directory. The poisoned pre-trained model can be found [here](https://drive.google.com/drive/folders/1HqBIbh8uPkgjASVgBqoVg7E1_8nXbqx-?usp=drive_link).

## Defense Baselines:
FT: You can keep fine-tuning the poisoned model on the clean dataset by comment prune code and attention loss in attention_normalization.py <br>
FTH: You can set a higher learning rate in config.py when you are running modified attention_normalization.py <br>
[MEFT](https://aclanthology.org/2023.findings-acl.237.pdf): They did not release code. I already provide the max entropy loss in util.py. Just use this loss to tune the model before you use cross-entropy loss to train the model.

## Demo 

**Rare Word Attack**

Without Defense

![image](https://github.com/user-attachments/assets/8e00baec-9ebc-47fa-9bd9-b476f072da1c)


Pure Results

![image](https://github.com/user-attachments/assets/c18fe7fc-361b-4a4c-9cbd-d170024c988b)


## Citation:
```bibtex
@InProceedings{pmlr-v235-zhao24r,
  title = 	 {Defense against Backdoor Attack on Pre-trained Language Models via Head Pruning and Attention Normalization},
  author =       {Zhao, Xingyi and Xu, Depeng and Yuan, Shuhan},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {61108--61120},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/zhao24r/zhao24r.pdf},
  url = 	 {https://proceedings.mlr.press/v235/zhao24r.html},
  abstract = 	 {Pre-trained language models (PLMs) are commonly used for various downstream natural language processing tasks via fine-tuning. However, recent studies have demonstrated that PLMs are vulnerable to backdoor attacks, which can mislabel poisoned samples to target outputs even after a vanilla fine-tuning process. The key challenge for defending against the backdoored PLMs is that end users who adopt the PLMs for their downstream tasks usually do not have any knowledge about the attacking strategies, such as triggers. To tackle this challenge, in this work, we propose a backdoor mitigation approach, PURE, via head pruning and normalization of attention weights. The idea is to prune the attention heads that are potentially affected by poisoned texts with only clean texts on hand and then further normalize the weights of remaining attention heads to mitigate the backdoor impacts. We conduct experiments to defend against various backdoor attacks on the classification task. The experimental results show the effectiveness of PURE in lowering the attack success rate without sacrificing the performance on clean texts.}
}






