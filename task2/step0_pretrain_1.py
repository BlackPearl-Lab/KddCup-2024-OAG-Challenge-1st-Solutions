from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import gc
import re
from tqdm.auto import tqdm
import ctypes
import torch.optim as optim
from os.path import join
import utils
from bs4 import BeautifulSoup
from collections import defaultdict as dd
from fuzzywuzzy import fuzz
import torch.distributed as dist
libc = ctypes.CDLL("libc.so.6")
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
from transformers import get_cosine_schedule_with_warmup, AdamW, get_linear_schedule_with_warmup, set_seed
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial

warnings.filterwarnings("ignore")
import torch.nn.functional as F
def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        pad_mask = pad_mask.bool()
        pad_mask = ~pad_mask
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from scipy.stats import bernoulli
import random
def setup_seed(seed):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    random.seed(seed)
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
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


class KddDataSet(Dataset):
    def __init__(self, text, labels, tokenizer, max_seq_len):
        self.text = text
        self.labels = labels
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        label = self.labels[item]
        text = text
        input_ids = self.tokenizer.encode(text,add_special_tokens=False)
        if len(input_ids) > (self.max_seq_len - 2):
            input_ids = input_ids[:(self.max_seq_len - 2)]
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        return input_ids, label

    @staticmethod
    def collate(batch):
        batch_input_ids, batch_labels = [], []
        for input_ids, label in batch:
            batch_input_ids.append(input_ids)
            batch_labels.append(label)
        max_seq_len = max([len(x) for x in batch_input_ids])
        pad_input_ids = []
        for i in range(len(batch_input_ids)):
            input_ids = batch_input_ids[i]
            if len(input_ids) < max_seq_len:
                input_ids = input_ids + [0] * (max_seq_len - len(input_ids))
            pad_input_ids.append(input_ids)
        batch_input_ids = torch.tensor(pad_input_ids, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        batch_attention_mask = (batch_input_ids > 0).long()
        return {'input_ids':batch_input_ids, 'attention_mask':batch_attention_mask, 'labels':batch_labels}


def build_optimizer_and_scheduler(model, num_total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    param_optimizer = []

    for name, param in model.named_parameters():
        param_optimizer.append((name, param))

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': 1e-5},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': 1e-5},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-6)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.05 * num_total_steps, num_training_steps=num_total_steps * 1.05
    )

    return optimizer, scheduler

def get_data():
    x_pos, x_neg = [], []
    y_pos, y_neg = [], []
    idx_pos, idx_neg = [], []
    papers = utils.load_json('./data', "paper_source_gen_by_rule.json")
    papers = sorted(papers, key=lambda x: x["_id"])

    papers_train = papers

    pids_train = {p["_id"] for p in papers_train}

    in_dir = join('./data', "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        if f.endswith(".xml"):
            files.append(f)

    pid_to_source_titles = dd(list)
    for paper in tqdm(papers):
        pid = paper["_id"]
        for ref in paper["refs_trace"]:
            pid_to_source_titles[pid].append(ref["title"].lower())

    total = []
    for cur_pid in tqdm(pids_train):
        f = open(join(in_dir, cur_pid + ".xml"), encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")

        source_titles = pid_to_source_titles[cur_pid]
        if len(source_titles) == 0:
            continue

        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            title = ""
            flag = False
            if ref.analytic is not None and ref.analytic.title is not None:
                title += ref.analytic.title.text.lower()
                flag = True
            if ref.monogr is not None and ref.monogr.title is not None and flag is False:
                title += ref.monogr.title.text.lower()
            bid_to_title[bid] = title
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx

        flag = False

        cur_pos_bib = set()
        for bid in bid_to_title:
            cur_ref_title = bid_to_title[bid]
            for label_title in source_titles:
                if fuzz.ratio(cur_ref_title, label_title) >= 80:
                    flag = True
                    cur_pos_bib.add(bid)

        cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib
        total.append(len(source_titles) - len(cur_pos_bib))
        if len(source_titles) > len(cur_pos_bib):
            a = 1
        if not flag:
            continue

        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue

        bib_to_contexts = utils.find_bib_context(xml)

        for bib in cur_pos_bib:
            cur_context = " ".join(bib_to_contexts[bib])
            x_pos.append(cur_context)
            y_pos.append(1)
            idx_pos.append(cur_pid)

        for bib in cur_neg_bib:
            cur_context = " ".join(bib_to_contexts[bib])
            x_neg.append(cur_context)
            y_neg.append(0)
            idx_neg.append(cur_pid)


    with open('./data/PST/rule_data_pos_bib_context_train.txt', 'w') as f:
        for item in x_pos:
            f.write(item.replace('\n', '') + '\n')
    with open('./data/PST/rule_data_neg_bib_context_train.txt', 'w') as f:
        for item in x_neg:
            f.write(item.replace('\n', '') + '\n')

def train():
    pos_cnt = 0
    train_texts = []
    train_labels = []
    with open('./data/PST/rule_data_pos_bib_context_train.txt', 'r', encoding='utf-8') as f:
        for line in f:
            pos_cnt += 1
            train_texts.append(line)
            train_labels.append(1)
    with open('./data/PST/rule_data_neg_bib_context_train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        lines = lines[:(pos_cnt * 10)]
        for line in lines:
            train_texts.append(line)
            train_labels.append(0)
    print('加载数据完成！')

    class_weight = len(train_labels) / (2 * np.bincount(train_labels))
    class_weight = torch.Tensor(class_weight).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)



    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    dataset = KddDataSet(train_texts, train_labels, tokenizer, 2048)
    sampler = DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate, sampler=sampler,num_workers=0)
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.update({'num_labels': 2, 'max_position_embeddings':2048})
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, config=config)
    state_dict = torch.load('./save_pretrain/model_save_deberta_large/pretrain_step50000.bin', map_location='cpu')  #一次预训练
    model.load_state_dict(state_dict,strict=False)

    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)
    epochs = 20
    num_total_steps = len(train_dataloader) * epochs
    optimizer, scheduler = build_optimizer_and_scheduler(model, num_total_steps)
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    step = 0
    for epoch in range(1, epochs+1):
        model.train()
        sampler.set_epoch(epoch)
        with tqdm(train_dataloader) as pBar:
            pBar.set_description(desc=f'[training]')
            for batch in pBar:
                step += 1
                with autocast():
                    for k in batch.keys():
                        batch[k] = batch[k].to(device)
                    logits= model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                    loss = criterion(logits, batch['labels'])
                scaler.scale(loss).backward()

                scaler.step(optimizer)
                optimizer.zero_grad()
                scheduler.step()
                scaler.update()
                pBar.set_postfix(loss=loss.item())
        if rank == 0:
            model.eval()
            torch.save(model.module.state_dict(), f'./out/grafting_learning_deberta_large/pretrain_epoch{epoch}.bin')
            break

if __name__ == '__main__':

    # This code is used to fintune Grafting learning model, so we can use 'paper_source_gen_by_rule.json' to get a model whose output probability is not very confident.

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    setup_seed(seed=42)
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    warnings.filterwarnings('ignore')
    os.makedirs('./out/grafting_learning_deberta_large', exist_ok=True)
    MODEL_PATH = 'microsoft/deberta-v3-large'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if rank == 0:
        get_data()
    train()
