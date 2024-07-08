
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import gc
import pandas as pd
import pickle
import sys
import numpy as np
from tqdm.autonotebook import trange
import json
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
    lora_path = sys.argv[3]
    doc_name_path = f"../train_features/{model_name}_doc.pkl"

    f = open(f"../data/AQA/qa_train.txt")
    dev_data = []
    for line in f.readlines():
        dev_data.append(eval(line))
    dev_data = pd.DataFrame(dev_data)
    dev_data['clear_body'] = dev_data['body'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
    dev_data['query_and_body'] = dev_data['question'] + "###body:" + dev_data['clear_body']
    dev_data['query_id'] = list(range(len(dev_data)))
    dev_data = dev_data.drop_duplicates("question")
    print(dev_data.shape)

    # dev_data = dev_data.sample(n=100,random_state=2025)

    task_description = 'Given a web search query and a relevant body, retrieve the title and abstract of papers that are pertinent to the query.'
    dev_data['query_and_body'] = dev_data['query_and_body'].apply(lambda x: get_detailed_instruct(
        task_description, ' '.join(x.split(' ')[:256])
    ))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    if lora_path !='none':
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

    # f = open(f"../sub_test/{model_name}.txt", 'w')
    # for pid in predicts_test:
    #     pid = ",".join(pid[:20])
    #     f.write(pid + "\n")
    # f.close()

    dev_data.to_parquet(f"../train_features/train_{model_name}_rank1000.parquet", index=False)

    dev_data['top_recall_pids'] = predicts_test
    dev_data['new_had_recall_pids'] = predicts_test


    train_data = dev_data
    def do(x, y):
        res = []
        for xi in x:
            if xi not in y:
                res.append(xi)
        return res

    train_data['new_had_recall_pids'] = list(
        map(lambda x, y: do(x, y), train_data['new_had_recall_pids'], train_data['pids']))

    with open("../data/AQA/pid_to_title_abs_new.json") as f:
        data = json.load(f)


    def do(x):
        res = []
        for xi in x:
            paper_info = data[xi]
            r = {"title": paper_info['title'], 'text': paper_info['abstract']}
            res.append(r)
        return res


    train_data['new_had_recall_ctxs'] = train_data['new_had_recall_pids'].apply(lambda x: do(x))
    train_data['new_positive_ctxs'] = train_data['pids'].apply(lambda x: do(x))

    # 只保存paper_id
    bge_train = []
    text_len = []
    cnt = 200

    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    task_description = 'Given a web search query and a relevant body, retrieve the title and abstract of papers that are pertinent to the query.'
    for _, row in train_data.iterrows():
        query = get_detailed_instruct(task_description, row['question'] + "###body:" + row['clear_body'])
        pos = []
        for data in row['pids']:
            pos.append(data)
            text_len.append(len(pos[-1]))
        neg = []
        hard_negative_ctxs = row['new_had_recall_pids'][:cnt]
        for data in hard_negative_ctxs:
            neg.append(data)
            text_len.append(len(neg[-1]))
        bge_train.append({'query': query, 'pos': pos, 'neg': neg})
    with open(f"../data/{model_name}_recall_top_{cnt}.jsonl", 'w') as f:
        json.dump(bge_train, f)

    cnt = 100
    bge_train = []
    for _,row in train_data.iterrows():
        query = row['question'] + "###" + row['clear_body']
        pos = []
        for data in row['new_positive_ctxs']:
            pos.append("###Paper title:"+str(data['title'])+"###Paper abstract:"+str(data['text']))
        neg = []
        hard_negative_ctxs = row['new_had_recall_ctxs'][:cnt]
        for data in hard_negative_ctxs:
            neg.append("###Paper title:"+str(data['title'])+"###Paper abstract:"+str(data['text']))
        bge_train.append({'query':query,'pos':pos,'neg':neg,'prompt':"Given a query with a relevant body,along with a title and abstract of paper,determine whether the paper is pertinent to the query by providing a prediction of either 'Yes' or 'No'."})

    with open(f"../data/{model_name}_recall_top_{cnt}_for_rank.jsonl", 'w') as f:
        json.dump(bge_train, f)