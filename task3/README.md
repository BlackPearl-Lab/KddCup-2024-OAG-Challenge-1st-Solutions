# OAG-AQA Rank 1 Solution

In this competition, we primarily employed hard example mining and iterative multi-path recall methods to achieve the final first-place result. The specific process is shown in the diagram below:

![Local Image](./assets/a.jpg "Local Sample Image")

We first obtained hard examples using an open-source vector model (model used: SFR-Embedding-Mistral), then fine-tuned them through contrastive learning, and finally recalled the top 100 for rerank training (model used: SOLAR-10.7B). This single process scored 0.201 in the final test set.

Furthermore, we adopted an iterative pipeline to obtain a diverse set of models and results (we iterated through 7 rounds in total), ultimately achieving a score of 0.223 through rank average fusion.

# Prediction Process

##  Prerequisites
```
Linux
GPU Resources:8 A100 80G GPUs
python 3.8+
torch 2.3.0+cu121 (It may also be compatible with some lower versions.)
pip install -r requirements_infer.txt
```
## step1.Download the LLM to the specified folder.
https://huggingface.co/Salesforce/SFR-Embedding-Mistral<br>
https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0

a.Place these two models in the ./model_save folder.

b.Download the LoRa parameters from here() and place them in the ./model_save folder.


## step2. Download the AQA-test-public.zip and extract it to the data folder.
```
cd ./data 
unzip  AQA-test-public.zip
```

## step3.Run the inference script.
```
cd ./inference
sh run.sh
```

The complete prediction process includes **7 models** and takes approximately **40** hours to complete on resources with 8 A100*80G cards.
The latest result file is located in the sub_test directory, named merge_7_model_last.txt.
# Train Process


