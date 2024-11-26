**PURE**

This file contains all scripts for the algorithm PURE. To run the PURE on poisoned model, you need to get the poisoned pretrain model by runing the script
pretrained_model_poisoning.py. The poisoned model will be saved in poisoned_model directory. Then you need to run attention_head_pruning.py to get the pruned
heads information and norm coefficients for each attention layer. At last, you need to run the attention_normalization.py to get the final results.

**Directory Setting**
- Folder A
  - File 1
  - File 2
  - Folder B
    - File 3
- Folder C
  - File 4
