**Overview**

We propose LocHy-GenRec, which enables group-level modeling and constructs hybrid identifiers to support generative recommendation with item SID.

<img width="1131" height="545" alt="模型图12 21" src="https://github.com/user-attachments/assets/8f325969-122f-4200-b7bc-5b7cd735e927" />


For representative studies on static identifier construction, please refer to the following works:
- [TIGER](https://github.com/EdoardoBotta/RQ-VAE-Recommender)
- [LETTER](https://github.com/HonghuiBao2000/LETTER)

## Requirement
```bash
torch==2.7.1+cu118 

accelerate

tokenizers

sentencepiece

deepspeed

evaluate

peft

bitsandbytes

tqdm

transformers...
```
**GENREC**


## Run GENREC
```bash
cd LocHy-GenRec
python -u ".../group_estimation.py"
python -u ".../hy_genrec.py"
```
**Resources**
Limited by GitHub file upload constraints, we offer a Baidu Netdisk link to download the full Checkpoints folder.
