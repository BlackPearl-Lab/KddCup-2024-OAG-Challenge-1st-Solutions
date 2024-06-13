
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import gc
import pandas as pd
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
)
import warnings
warnings.filterwarnings('ignore')

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


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def inference(df, model, tokenizer, device):
    batch_size = 32
    max_length = 256
    sentences = list(df['query_and_body'].values)
    pids = list(df['query_id'].values)
    all_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=False):
        sentences_batch = sentences_sorted[start_index: start_index + batch_size]
        features = tokenizer(sentences_batch, max_length=max_length, padding=True, truncation=True,
                             return_tensors="pt")
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


if __name__ == '__main__':
    device = 'cuda:0'
    model_name = sys.argv[1]
    model_path = sys.argv[2]
    doc_name_path = f"../test_features/{model_name}_doc.pkl"
    lora_path = f"../model_save/{model_name}/epoch_19_model/adapter.bin"

    f = open(f"../data/AQA-test-public/qa_test_wo_ans_new.txt")
    dev_data = []
    for line in f.readlines():
        dev_data.append(eval(line))
    dev_data = pd.DataFrame(dev_data)
    dev_data['clear_body'] = dev_data['body'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
    dev_data['query_and_body'] = dev_data['question'] + "###body:" + dev_data['clear_body']
    dev_data['query_id'] = list(range(len(dev_data)))

    # dev_data = dev_data.sample(n=100,random_state=2025)

    task_description = 'Given a web search query and a relevant body, retrieve the title and abstract of papers that are pertinent to the query.'
    dev_data['query_and_body'] = dev_data['query_and_body'].apply(lambda x: get_detailed_instruct(
        task_description, ' '.join(x.split(' ')[:256])
    ))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
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

    query_embedding_dict = inference(dev_data, model, tokenizer, device)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    with open(f"{doc_name_path}", 'rb') as f:
        paper_embeddings = pickle.load(f)
    sentence_embeddings = np.concatenate([e.reshape(1, -1) for e in list(paper_embeddings.values())])

    index_paper_text_embeddings_index = {index: paper_id for index, paper_id in
                                         enumerate(list(paper_embeddings.keys()))}

    sentence_embeddings_tensor = torch.tensor(sentence_embeddings).to(device)

    predicts_test = []
    for _, row in tqdm(dev_data.iterrows()):
        query_id = row['query_id']
        query_em = query_embedding_dict[query_id].reshape(1, -1)
        query_em = torch.tensor(query_em).to(device).view(1, -1)
        score = F.cosine_similarity(query_em, sentence_embeddings_tensor)
        sort_index = torch.sort(-score).indices.detach().cpu().numpy().tolist()[:1000]
        pids = [index_paper_text_embeddings_index[index] for index in sort_index]
        predicts_test.append(pids)

    dev_data['top_recall_pids'] = predicts_test

    f = open(f"../sub_test/{model_name}.txt", 'w')
    for pid in predicts_test:
        pid = ",".join(pid[:20])
        f.write(pid + "\n")
    f.close()

    dev_data.to_parquet(f"../sub_test/{model_name}_rank1000.parquet", index=False)
