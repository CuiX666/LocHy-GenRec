

## Overview

We propose LocHy-GenRec, which enables group-level modeling and constructs hybrid identifiers to support generative recommendation with item SID.
<img width="2102" height="1080" alt="模型图26 6 5" src="https://github.com/user-attachments/assets/65773800-17cb-444b-ad38-eba0e607e6d0" />


## Base
For representative studies on static identifier construction, please refer to the following works:
- [RQ-VAE-Recommendation](https://github.com/EdoardoBotta/RQ-VAE-Recommender)(Recommender Systems with Generative Retrieval)
- [LETTER](https://github.com/HonghuiBao2000/LETTER)(Learnable Item Tokenization for Generative Recommendation)

The required LLM can be downloaded via the [Meta](https://developer.meta.com/ai/).

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

## Run GENREC
```bash
cd LocHy-GenRec
python -u ".../group_estimation.py"
python -u ".../hy_genrec.py"
```
## Resources
```bash
Limited by GitHub file upload constraints, we offer a Baidu Netdisk link to download the full Checkpoints folder.
```
