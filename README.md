# You Only Prompt Once: On the Capabilities of Prompt Learning on Large Language Models to Tackle Toxic Content

This is the official implementation of our paper [You Only Prompt Once: On the Capabilities of Prompt Learning on Large Language Models to Tackle Toxic Content](https://arxiv.org/abs/2308.05596).

## Environment Setup

```
conda env create --file environment.yaml &&
conda activate toxic_prompt
```

## Datasets

Datasets we used in this paper are in `parsed_dataset` folder.
```
# Task 1
Dataset: ["HateXplain", "USElectionHate20", "HateCheck", "SBIC.v2", "measuring-hate-speech"]

# Task 2:
Dataset: ["TSD"]

# Task 3:
Dataset: ["Parallel", "Paradetox"]
```


## Task 1: Toxicity Classification

```
# Train the prompt tuning model. 
python 1_toxicity_classification.py --plm_eval_mode --model t5 --model_name_or_path t5-small --dataset HateXplain
```


## Task 2: Toxic Span Detection

Train the prompt and perform the generation task:
```
# Example: TSD dataset with t5-small model.
python 2_and_3_toxic_generation.py --plm_eval_mode --model t5 --model_name_or_path t5-small --dataset TSD
```
Evaluation:
```
python 2_calculate_span.py --file_path sfs_out/task23/TSD_t5-small_True.txt
```

For the baseline methods, we follow the implementation from [toxic-span](https://github.com/ipavlopoulos/toxic_spans)

## Task 3: Detoxification

Note that Task 2 and Task 3 share the same code as they are both generation tasks.

Train the prompt and perform the generation task:
```
# Example: Parallel dataset with t5-small model.
python 2_and_3_toxic_generation.py --plm_eval_mode --model t5 --model_name_or_path t5-small --dataset Parallel
```

Evaluation for toxicity change:
```
python 3_perspective_evaluation.py --file_path sfs_out/task23/Parallel_t5-small_True.txt --key YOUR_PERSPECTIVE_API_KEY
```

Note that for the baseline methods and evaluation regarding other metrics (e.g., BLEU, SIM, PPL), we follow the implementation of [paradetox](https://github.com/s-nlp/paradetox).

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{HZSZ224,
author = {Xinlei He and Savvas Zannettou and Yun Shen and Yang Zhang},
title = {{You Only Prompt Once: On the Capabilities of Prompt Learning on Large Language Models to Tackle Toxic Content}},
booktitle = {{IEEE Symposium on Security and Privacy (S\&P)}},
publisher = {IEEE},
year = {2024}
}
```