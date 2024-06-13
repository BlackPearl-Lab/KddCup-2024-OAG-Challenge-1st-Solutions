#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import json
import pickle
from bs4 import BeautifulSoup
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import os
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import json
import pickle
# from sentence_transformers import SentenceTransformer
import sys
import itertools
import os
import pickle
import sys
import numpy as np
from tqdm.autonotebook import trange

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,
)


def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# In[ ]:


save_name = sys.argv[1]
path_pre = sys.argv[2]
model_path = sys.argv[3]
lora_path = sys.argv[4]
num_gpu = int(sys.argv[5])
# num_gpu = 8
print("path_pre:", path_pre)
print("model_path:", model_path)
print("lora_path:", lora_path)
print("save_name:", save_name)
print("num_gpu:", num_gpu)

# In[ ]:


with open(f"{path_pre}../data/AQA/pid_to_title_abs_new.json") as f:
    data = json.load(f)


def get_paper_info_text(pid_to_info_all):
    res = {}
    for paper_id, paper_info in pid_to_info_all.items():
        text = "###Paper title:" + str(paper_info['title']) + "###Paper abstract:" + str(paper_info['abstract'])
        res[paper_id] = text.strip()
    return res


data_dict = get_paper_info_text(data)
print(len(data_dict))

# In[ ]:


# In[ ]:


df = pd.DataFrame({
    "pid": list(data_dict.keys()),
    "passages": list(data_dict.values()),
})

# df = df.sample(n=1000,random_state=2025)


# In[ ]:


print(df['passages'].apply(lambda x: len(x.split(' '))).describe([0.9, 0.99]))

use_device = [f'cuda:{i}' for i in range(num_gpu)]
from threading import Thread

models = []
for device in use_device:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    if lora_path!="none":
        config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        d = torch.load(lora_path, map_location=model.device)
        model.load_state_dict(d, strict=False)
        model = model.merge_and_unload()
    model = model.eval()
    model = model.to(device)
    models.append((model, tokenizer))


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


def inference(df, model, tokenizer, device):
    batch_size = 16
    max_length = 256
    sentences = list(df['passages'].values)
    pids = list(df['pid'].values)
    all_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=False):
        sentences_batch = sentences_sorted[start_index: start_index + batch_size]
        features = tokenizer(sentences_batch, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        features = batch_to_device(features, device)
        with torch.no_grad():
            outputs = model(**features)
            embeddings = last_token_pool(outputs.last_hidden_state, features['attention_mask'])
            embeddings = embeddings.detach().cpu().numpy().tolist()
        all_embeddings.extend(embeddings)

    all_embeddings = [np.array(all_embeddings[idx]).reshape(1, -1) for idx in np.argsort(length_sorted_idx)]

    sentence_embeddings = np.concatenate(all_embeddings, axis=0)
    result = {pids[i]: em for i, em in enumerate(sentence_embeddings)}
    return result


results = {}


def run_inference(df, model, index, device):
    results.update(inference(df, model[0], model[1], device))


ts = []
# df = df.sample(n=300,random_state=2023)
df['fold'] = list(range(len(df)))
df['fold'] = df['fold'] % len(use_device)
print(df['fold'].value_counts())
for index, device in enumerate(use_device):
    t0 = Thread(target=run_inference, args=(df[df['fold'] == index], models[index], index, device))
    ts.append(t0)
for i in range(len(ts)):
    ts[i].start()
for i in range(len(ts)):
    ts[i].join()

# In[ ]:

print(len(results))
with open(f"{path_pre}../test_features/{save_name}.pkl", 'wb') as f:
    pickle.dump(results, f)

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:




