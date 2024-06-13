import math
import json
import glob
import collections
import random
from pathlib import Path
import pandas as pd
import numpy as np
import os
import copy

from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm
import pickle
import gc
import utils
import faiss
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
import torch
from prefetch_generator import BackgroundGenerator
from functools import partial

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from transformers import AdamW, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGM():
    '''
    Example
    # 初始化
    fgm = FGM(model,epsilon=1,emb_name='word_embeddings.')
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack() # 在embedding上添加对抗扰动
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
    '''

    def __init__(self, model, emb_name='word_embeddings.', epsilon=0.25):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 2020
seed_everything(SEED)

DATA_PATH = "./data/"
BERT_PATH = "/home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/bge-m3"
PROMPT_LEN = 1024
WIKI_LEN = 1024
MAX_LEN = 1024
BATCH_SIZE = 96
DEVICE = 'cuda'
use_hard_sample = False
K = 1
import logging

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


class LLMRecallDataSet(torch.utils.data.Dataset):
    def __init__(self, data):
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, use_fast=True)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        query = self.data[index]
        query_id = self.tokenizer.encode(query, add_special_tokens=False)
        if len(query_id) > 1022:
            query_id = [0] + query_id[:1022] + [2]
        else:
            query_id = [0] + query_id + [2]
        return query_id

    def collate_fn(self, batch):
        def sequence_padding(inputs, length=None, padding=1):
            """
            Numpy函数，将序列padding到同一长度
            """
            if length is None:
                length = max([len(x) for x in inputs])

            pad_width = [(0, 0) for _ in np.shape(inputs[0])]
            outputs = []
            for x in inputs:
                x = x[:length]
                pad_width[0] = (0, length - len(x))
                x = np.pad(x, pad_width, 'constant', constant_values=padding)
                outputs.append(x)

            return np.array(outputs, dtype='int64')

        batch_query, batch_answer = [], []
        for item in batch:
            query = item
            batch_query.append(query)

        batch_query = torch.tensor(sequence_padding(batch_query), dtype=torch.long)
        return batch_query


class TitleDataSet(torch.utils.data.Dataset):
    def __init__(self, data):
        self.tokenizer = AutoTokenizer.from_pretrained('/home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/deberta-v3-large', use_fast=True)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        paper = self.data[index]['paper_title'] + self.data[index]['paper_abstract']
        source = self.data[index]['source_title'] + self.data[index]['source_abstract']
        paper_id = self.tokenizer.encode(paper, add_special_tokens=False)
        source_id = self.tokenizer.encode(source, add_special_tokens=False)
        if len(paper_id) > 1022:
            paper_id = [1] + paper_id[:1022] + [2]
        else:
            paper_id = [1] + paper_id + [2]
        if len(source_id) > 1022:
            source_id = [1] + source_id[:1022] + [2]
        else:
            source_id = [1] + source_id + [2]
        return paper_id, source_id, self.data[index]['label'], self.data[index]['idx']

    def collate_fn(self, batch):
        def sequence_padding(inputs, length=None, padding=0):
            """
            Numpy函数，将序列padding到同一长度
            """
            if length is None:
                length = max([len(x) for x in inputs])

            pad_width = [(0, 0) for _ in np.shape(inputs[0])]
            outputs = []
            for x in inputs:
                x = x[:length]
                pad_width[0] = (0, length - len(x))
                x = np.pad(x, pad_width, 'constant', constant_values=padding)
                outputs.append(x)

            return np.array(outputs, dtype='int64')

        batch_paper, batch_source, batch_label, batch_idx = [], [], [], []
        for paper, source, label, idx in batch:
            batch_paper.append(paper)
            batch_source.append(source)
            batch_label.append(label)
            batch_idx.append(idx)

        batch_paper = torch.tensor(sequence_padding(batch_paper), dtype=torch.long)
        batch_source = torch.tensor(sequence_padding(batch_source), dtype=torch.long)
        batch_label = torch.tensor(batch_label, dtype=torch.long)

        return {'paper':batch_paper, 'source':batch_source, 'label':batch_label, 'idx':batch_idx}


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_loader(prompt, batch_size, train_mode=True, num_workers=4):
    ds_df = LLMRecallDataSet(prompt)
    dataloader_class = partial(DataLoader, pin_memory=True)
    loader = dataloader_class(ds_df, batch_size=batch_size, shuffle=train_mode, collate_fn=ds_df.collate_fn,
                              num_workers=num_workers)
    return loader



class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class RecallModel(nn.Module):
    def __init__(self):
        super(RecallModel, self).__init__()
        self.bert_model = AutoModel.from_pretrained(BERT_PATH)
        self.mean_pooler = MeanPooling()

    def forward(self, input_ids):
        attention_mask = input_ids != 1
        out = self.bert_model(input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.mean_pooler(out, attention_mask)

        return x

class TitleModel(nn.Module):
    def __init__(self):
        super(TitleModel, self).__init__()
        self.deberta = AutoModel.from_pretrained('/home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/deberta-v3-large')
        self.linear = nn.Linear(1024, 2)

    def forward(self, paper, source):
        paper_attention_mask = (paper != 0)
        source_attention_mask = (source != 0)
        paper_embed = self.deberta(input_ids=paper, attention_mask=paper_attention_mask).last_hidden_state.mean(1)
        source_embed = self.deberta(input_ids=source, attention_mask=source_attention_mask).last_hidden_state.mean(1)
        final_embed = (paper_embed + source_embed) / 2
        logits = self.linear(final_embed)
        return paper_embed, source_embed, logits


def SimCSE_loss(topic_pred, content_pred, hards=None, tau=0.05):
    if use_hard_sample:
        similarities = F.cosine_similarity(topic_pred.unsqueeze(1), content_pred.unsqueeze(0), dim=2)  # B,B
        hard_sims = faiss.IndexFlatIP(1024)
        for i in range(K):
            hard_sim = F.cosine_similarity(topic_pred, hards[i])
            hard_sim = hard_sim.unsqueeze(dim=1)
            hard_sims.append(hard_sim)
        hard_sims = torch.concat(hard_sims, axis=1)
        similarities = torch.concat([similarities, hard_sims], axis=1)
        y_true = torch.arange(0, topic_pred.size(0)).to(DEVICE)
        # similarities = similarities - torch.eye(pred.shape[0]) * 1e12
        similarities = similarities / tau
        loss = F.cross_entropy(similarities, y_true)
    else:
        similarities = F.cosine_similarity(topic_pred.unsqueeze(1), content_pred.unsqueeze(0), dim=2)  # B,B
        y_true = torch.arange(0, topic_pred.size(0)).to(DEVICE)
        # similarities = similarities - torch.eye(pred.shape[0]) * 1e12
        similarities = similarities / tau
        loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


from torch.cuda.amp import autocast, GradScaler
import faiss
def eval_all_title(model, dataloader):
    index = faiss.IndexFlatIP(1024)
    for batch in tqdm(dataloader):
        with torch.no_grad():
            batch = batch.to(DEVICE)
            with autocast():
                output = model(batch).cpu().detach().numpy()
            faiss.normalize_L2(output)
            index.add(output)
    faiss.write_index(index, './data/all_title_alltext.bin')


from os.path import join
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz


def evalaute(text, model):
    new = []
    for item in text:
        if item == '':
            item = ' '
        new.append(item)
    dataloader = get_loader(new, 128, False, 8)
    res = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            batch = batch.to(DEVICE)
            with autocast():
                output = model(batch).cpu().detach().numpy()
            faiss.normalize_L2(output)
            res.append(output)
    res = np.concatenate(res, axis=0)
    return res



def get_test_abstract_addcite(model):
    data1 = pd.read_pickle('./data/llm_final_title_addbib.pickle')
    index = faiss.read_index('./data/all_title.bin')
    print("加载完成 迁移到gpu中")
    index = faiss.index_cpu_to_all_gpus(index)
    all_title = data1['ref_title'].tolist()
    title_embed = evalaute(all_title, model)
    n_blocks = int(np.ceil(title_embed.shape[0] / 50))
    idx_list = []
    for i in range(n_blocks):
        # 计算当前块的起始和结束索引
        start_index = i * 50
        end_index = min((i + 1) * 50, title_embed.shape[0])
        block = title_embed[start_index:end_index]

        _, idx = index.search(block, 1)
        idx_list.append(idx)
    ref_info = []
    idx = np.concatenate(idx_list, axis=0)
    tt_title = []
    ttt_title = []
    for i, id in enumerate(idx):
        origin = data[id[0]]
        tmp = {}
        for k in ['abstract', 'year', 'authors', 'venue', 'n_citation', 'keywords', 'venue_cite']:
            tmp['ref_' + k] = origin[k]
        tt_title.append(origin['title'])
        ref_info.append(tmp)
    # index = faiss.read_index('./data/all_title.bin')
    # print("加载完成 迁移到gpu中")
    # index = faiss.index_cpu_to_all_gpus(index)
    all_title = data1['source_title'].tolist()
    title_embed = evalaute(all_title, model)
    n_blocks = int(np.ceil(title_embed.shape[0] / 50))
    idx_list = []
    for i in range(n_blocks):
        # 计算当前块的起始和结束索引
        start_index = i * 50
        end_index = min((i + 1) * 50, title_embed.shape[0])
        block = title_embed[start_index:end_index]

        _, idx = index.search(block, 1)
        idx_list.append(idx)
    source_info = []
    idx = np.concatenate(idx_list, axis=0)
    for i, id in enumerate(idx):
        origin = data[id[0]]
        tmp = {}
        for k in ['abstract', 'year', 'authors', 'venue', 'n_citation', 'keywords', 'venue_cite']:
            tmp['source_' + k] = origin[k]
        source_info.append(tmp)
        ttt_title.append(origin['title'])

    df1 = pd.DataFrame(source_info)
    df2 = pd.DataFrame(ref_info)
    data1 = pd.concat([data1, df1, df2], axis=1)
    data1.to_pickle('./data/llm_final_title_addbib_addcite_moreinfo.pickle')

def train_title_abstract_addcite(model):
    data1 = pd.read_pickle(f'./data/llm_notsample_with_allinfo_addbib_addcite.pickle')
    index = faiss.read_index('./data/all_title.bin')
    print("加载完成 迁移到gpu中")
    index = faiss.index_cpu_to_all_gpus(index)
    all_title = data1['ref_title'].tolist()
    title_embed = evalaute(all_title, model)
    n_blocks = int(np.ceil(title_embed.shape[0] / 50))
    idx_list = []
    for i in range(n_blocks):
        # 计算当前块的起始和结束索引
        start_index = i * 50
        end_index = min((i + 1) * 50, title_embed.shape[0])
        block = title_embed[start_index:end_index]

        _, idx = index.search(block, 1)
        idx_list.append(idx)
    ref_info = []
    idx = np.concatenate(idx_list, axis=0)
    tt_title = []
    ttt_title = []
    for i, id in enumerate(idx):
        origin = data[id[0]]
        tmp = {}
        for k in ['abstract', 'year', 'authors', 'venue', 'n_citation', 'keywords', 'venue_cite']:
            tmp['ref_' + k] = origin[k]
        tt_title.append(origin['title'])
        ref_info.append(tmp)
    all_title = data1['source_title'].tolist()
    title_embed = evalaute(all_title, model)
    n_blocks = int(np.ceil(title_embed.shape[0] / 50))
    idx_list = []
    for i in range(n_blocks):
        # 计算当前块的起始和结束索引
        start_index = i * 50
        end_index = min((i + 1) * 50, title_embed.shape[0])
        block = title_embed[start_index:end_index]

        _, idx = index.search(block, 1)
        idx_list.append(idx)
    source_info = []
    idx = np.concatenate(idx_list, axis=0)
    for i, id in enumerate(idx):
        origin = data[id[0]]
        tmp = {}
        for k in ['abstract', 'year', 'authors', 'venue', 'n_citation', 'keywords', 'venue_cite']:
            tmp['source_' + k] = origin[k]
        source_info.append(tmp)
        ttt_title.append(origin['title'])
    data1 = data1[['idx', 'pred', 'bib', 'source_title', 'ref_title', 'occur_cnt']]
    df1 = pd.DataFrame(source_info)
    df2 = pd.DataFrame(ref_info)
    data1 = pd.concat([data1, df1, df2], axis=1)
    data1.to_pickle('./data/llm_notsample_with_allinfo_addbib_addcite_moreinfo.pickle')
import pickle as pkl
def get_test_abstract_mistral():
    data1 = pd.read_pickle('./data/llm_final_title.pickle')
    data = []
    with open('../OAG_data/new_DBLP.json', 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line.strip('\n')))
    # data.append({'title': '', 'abstract': '', 'keywords': [], 'year': '', 'authors': [], 'venue': '', 'n_citation': 0,
    #              'venue_cite': 0})
    paper_cut = np.array_split(data, 8)
    title_embed = []
    for i in range(8):
        with open(f'../track1/test_embed_{i}.pkl', 'rb') as f:
            title_embed.append(pkl.load(f))
    title_embed = np.concatenate(title_embed, axis=0)
    paper_embed = []
    for i in range(8):
        with open(f'../track1/paper(title_year_keywords_authors_venue)_embedding(SFR-Embedding-Mistral)_{i}.pkl', 'rb') as f:
            paper_embed.append(pkl.load(f))
    paper_index = []
    for i in range(8):
        index = faiss.IndexFlatIP(4096)
        index.add(paper_embed[i])
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, i, index)
        paper_index.append(index)

    n_blocks = int(np.ceil(title_embed.shape[0] / 10))
    ref_info = []
    for i in tqdm(range(n_blocks)):
        # 计算当前块的起始和结束索引
        start_index = i * 10
        end_index = min((i + 1) * 10, title_embed.shape[0])
        block = title_embed[start_index:end_index]
        total_sim, total_idx = [], []
        total_paper_idx = []
        for j in range(8):
            sim, idx = paper_index[j].search(block, 1)
            total_sim.append(sim)
            total_idx.append(idx)
            total_paper_idx.append([j] * sim.shape[0])
        total_sim = np.concatenate(total_sim, axis=1)
        total_idx = np.concatenate(total_idx, axis=1)
        sorted_indices = np.argsort(-total_sim, axis=1)
        total_paper_idx = np.array(total_paper_idx).T
        total_idx = np.take_along_axis(total_idx, sorted_indices, axis=1)
        total_paper_idx = np.take_along_axis(total_paper_idx, sorted_indices, axis=1)
        total_idx = list(total_idx[:, 0])
        total_paper_idx = list(total_paper_idx[:, 0])
        for j in range(len(total_idx)):
            origin = paper_cut[total_paper_idx[j]][total_idx[j]]
            tmp = {}
            for k in ['abstract', 'year', 'authors', 'venue', 'n_citation', 'keywords', 'venue_cite']:
                tmp['ref_' + k] = origin[k]
            ref_info.append(tmp)

    data2 = pd.DataFrame(ref_info)
    data2.to_pickle('./data/llm_final_onlyref_mistral.pickle')

def get_moreinfo():
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
    return new_DBLP


def process_data_addcite(now_data, tokenizer):
    a_text = now_data['text']
    blocks = a_text.split('\n')
    a_text = '\n'.join(blocks[:-4])
    a_last = '\n'.join(blocks[-4:])
    authors = now_data['ref_authors'][:3]
    author_cites = [x['author_cite'] if x['author_cite'] != 81768982 else 0 for x in authors]
    org_cites = [x['org_cite'] if x['org_cite'] != 122669414 else 0 for x in authors]
    org = [x['org'] for x in authors]
    author_name = [x['name'] for x in authors]
    tra_text = f"the year of source paper is {now_data['source_year']}, the keywords of source paper is {''.join(now_data['source_keywords'])}, the venue of source paper is {now_data['source_venue']}, the cite of source paper is {now_data['source_n_citation']}, the cite of source paper's venue is {now_data['source_venue_cite']},"
    if len(authors) == 0:
        tra_text += "the authors of this reference paper is None,"
    else:
        for i in range(len(authors)):
            idx = ['first', 'second', 'third'][i]
            tra_text += f"this reference paper's {idx} author's name is {author_name[i]}, total cite is {author_cites[i]}, organization is {org[i]}, total cite of this organization is {org_cites[i]},"

    tra_text += f"the year of this reference paper is {now_data['ref_year']}, the venue of this reference paper is {now_data['ref_venue']}, the keywords of this reference paper is {' '.join(now_data['ref_keywords'])}, the cite of this refenence paper is {now_data['ref_n_citation']}, the cite of reference paper's venue is {now_data['ref_venue_cite']}, the appearance time of this reference paper is {now_data['occur_cnt']}, the abstract of this reference paper is:{now_data['ref_abstract']}."
    text_ids = tokenizer(
        [tra_text], add_special_tokens=False,
        return_tensors='pt')['input_ids'][0][:2000]
    tra_text = tokenizer.decode(text_ids)
    b_text = f"\nIn addition, " + tra_text
    a_ids = tokenizer.encode(a_text, add_special_tokens=False)
    b_ids = tokenizer.encode(b_text, add_special_tokens=False)

    if len(a_ids) + len(b_ids) < (32400 - 100):
        a_text = tokenizer.decode(a_ids, skip_special_tokens=False) + a_last
        b_text = tokenizer.decode(b_ids, skip_special_tokens=False)
    else:
        a_ids = a_ids[:(32400 - len(b_ids) - 100)]
        a_text = tokenizer.decode(a_ids, skip_special_tokens=False) + a_last
        b_text = tokenizer.decode(b_ids, skip_special_tokens=False)
    text = a_text + b_text
    return text

def process_data_wrapper_addcite(args):
    data, tokenizer = args
    return process_data_addcite(data, tokenizer)
import multiprocessing

def prepare_llm_moreinfo_multiprocess_addcite(path):
    tokenizer = AutoTokenizer.from_pretrained('ZhipuAI/chatglm3-6b-32k', trust_remote_code=True)
    data = pd.read_pickle(f'./data/{path}.pickle')
    pool = multiprocessing.Pool(50)
    process_text = list(tqdm(pool.imap(process_data_wrapper_addcite, [(data.loc[index], tokenizer) for index in tqdm(range(len(data)))]),total=len(data)))
    pool.close()
    pool.join()
    data['process_text'] = process_text
    data.to_pickle(f'./data/{path}_processtext.pickle')

if __name__ == '__main__':
    data = get_moreinfo()
    data.append({'title':'', 'abstract':'', 'keywords':[], 'year':'', 'authors':[], 'venue':'', 'n_citation':0, 'venue_cite':0})
    print('加载数据完成')
    train = pd.DataFrame(data)
    dataloader = get_loader(train['title'].tolist(),
                                  batch_size=1000,
                                  train_mode=False,
                                  num_workers=8)
    model = RecallModel().to(DEVICE)
    model = torch.nn.parallel.DataParallel(model)
    model.load_state_dict(torch.load('./out/bge_m3_simcse_pretrain.bin',map_location='cpu'))
    eval_all_title(model, dataloader)
    train_title_abstract_addcite(model)
    #最终训练文件生成
    prepare_llm_moreinfo_multiprocess_addcite(path='llm_notsample_with_allinfo_addbib_addcite_moreinfo')
    #推理
    get_test_abstract_addcite(model)
    # get_test_abstract_mistral()

    #最终推理文件生成
    prepare_llm_moreinfo_multiprocess_addcite(path='llm_final_title_addbib_addcite_moreinfo')