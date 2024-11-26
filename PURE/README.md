**PURE**

This file contains all scripts for the algorithm PURE. To run the PURE on the poisoned model, you need to get the poisoned pre-train model by running the script
pretrained_model_poisoning.py. The poisoned model will be saved in the poisoned_model directory. Then you need to run attention_head_pruning.py to get the pruned
heads information and norm coefficients for each attention layer. At last, you need to run the attention_normalization.py to get the final results.

**Directory Setting**
  - PURE
    - head_coefficients
    - poisoned_model
      - BadNet
      - HiddenKiller
      - LayerWise
      - RIPPLe
      - StyleBkd
    - pretrained_model_poisoning.py
    - attention_head_pruning.py
    - attention_normalization.py
    - config.py
    - util.py
    - run_all.py
