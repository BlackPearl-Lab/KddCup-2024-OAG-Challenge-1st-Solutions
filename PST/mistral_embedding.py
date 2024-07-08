import argparse
import json
import pickle
import pandas as pd
parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
parser.add_argument('--gpu', default=0)
args = parser.parse_args()
args.gpu = int(args.gpu)

data1 = pd.read_pickle('./data/llm_final_title.pickle')
ref_title = data1['ref_title'].tolist()
import numpy as np
cut = np.array_split(ref_title, 8)
cut = cut[args.gpu]


prompt = "Given the title, year, keywords, author information, and journal information of a paper, recall the original text of the paper."

def get_paper_info_text(cut):
    res = []
    for paper_info in cut:
        text = prompt + paper_info
        res.append(text)
    return res
paper_text_dict = get_paper_info_text(cut)


from sentence_transformers import SentenceTransformer
sim_model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")


sentence_embeddings = sim_model.encode(paper_text_dict,batch_size=6,convert_to_numpy=True,show_progress_bar=True)


with open(f"./data/test_embed_{args.gpu}.pkl",'wb') as f:
    pickle.dump(sentence_embeddings,f)



data = []
print('开始加载数据')
with open('./OAG_data/DBLP-Citation-network-V15.1.json', 'r') as f:
    for line in f.readlines():
        data.append(json.loads(line))
from tqdm import tqdm
id2paper = {}
author2paper = {}
author2cite = {}
org2cite = {}
author2org = {}
venue2cite = {}
paper2venue = {}
for item in tqdm(data):
    id2paper[item['id']] = item
    n_citation = item['n_citation']
    for x in item['authors']:
        if x['id'] not in author2paper.keys():
            author2paper[x['id']] = [item]
        else:
            author2paper[x['id']].append(item)
        org = x['org']
        if org not in org2cite.keys():
            org2cite[org] = n_citation
        else:
            org2cite[org] += n_citation
        if x['id'] not in author2org.keys():
            author2org[x['id']] = set()
            author2org[x['id']].add(org)
        else:
            author2org[x['id']].add(org)
        if x['id'] not in author2cite.keys():
            author2cite[x['id']] = n_citation
        else:
            author2cite[x['id']] += n_citation
    venue = item['venue']
    if venue not in venue2cite.keys():
        venue2cite[venue] = n_citation
    else:
        venue2cite[venue] += n_citation
    paper2venue[item['id']] = venue
new_DBLP = []
from copy import deepcopy
for item in tqdm(data):
    new_item = deepcopy(item)
    new_item['venue_cite'] = venue2cite[new_item['venue']]
    for x in new_item['authors']:
        x['author_cite'] = author2cite[x['id']]
        x['org_cite'] = org2cite[x['org']]
    new_DBLP.append(new_item)

cut = np.array_split(new_DBLP, 8)
cut = cut[args.gpu]


prompt = "Given the title, year, keywords, author information, and journal information of a paper, recall the original text of the paper."

def get_paper_info_text(cut):
    res = []
    for paper_info in cut:
        text = prompt + 'title:' + paper_info['title']
        res.append(text)
    return res
paper_text_dict = get_paper_info_text(cut)

sentence_embeddings = sim_model.encode(paper_text_dict,batch_size=6,convert_to_numpy=True,show_progress_bar=True)


with open(f"./data/all_paper_embed_{args.gpu}.pkl",'wb') as f:
    pickle.dump(sentence_embeddings,f)