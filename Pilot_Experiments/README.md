**Directory Structure**

When you run these code, you can set the directory structure like:
  - Pilot Experiments
    - imdb
    - model
      - poisoned-bert-base
      - user_finetune
        - bert_model
        - classifier

**File Function**

1. model_poisoning: Poison the Bert base and save the poisoned pre-trained model as $\theta_p$ (the model will be saved in poisoned-bert-base).
2. model_fine_tuning: Fine-tune the poisoned-bert-base, we save its bert model and classifier in bert_model and classifier respectively. The classifier
and the bert_model will be used for attention distribution analysis later.
3. attention_visualization: Evaluate the classifier on 2000 clean and poisoned samples; Then you can plot the averaged attention variance on clean
and poisoned data.
4. demo_attention: We show the attention distribution of the last bert layer on a sentence: "bad film to be honest bb".
5. You can run the Python file **run_all.py** after you set the directory well. 

**Dataset**

The clean and poisoned dataset can be found here. [imdb](https://drive.google.com/drive/folders/1kG9UGuTNwwx6Wga61huz3lboSELQ4Zen?usp=sharing)

**Model**

The poisoned model and classifier can be found here. [model](https://drive.google.com/drive/folders/1NDZ8mU6OQqlFqhmhy_o2nSm4RzyF6t4g?usp=sharing)

**Attention Variance heat map on 2000 samples**

![pilot experiments](https://github.com/user-attachments/assets/e1dd3f10-c8b3-4c5d-a3ce-887f03446bd9)
