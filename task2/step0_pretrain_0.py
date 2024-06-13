from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import gc
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
import ctypes
import torch.optim as optim
import torch.distributed as dist
libc = ctypes.CDLL("libc.so.6")
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
from transformers import get_cosine_schedule_with_warmup, AdamW, get_linear_schedule_with_warmup, set_seed
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
import json
from tqdm import tqdm
import pandas as pd
warnings.filterwarnings("ignore")

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


class BertDataSet_MLM(Dataset):
    def __init__(self, MODEL_PATH, data):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.mask_token = self.tokenizer.mask_token_id
        self.special_token = self.tokenizer.all_special_ids
        self.probability_list = bernoulli.rvs(p=0.2, size=len(data))  # 20%概率进行替换

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.loc[index]
        text = f"title:{item['title']}\nabstract:{item['abstract']}\nkeywords:{','.join(item['keywords'])}\nyear:{item['year']}\nvenue:{item['venue']}"
        # item = self.data[index]
        return self.encode(text)

    def ngram_mask(self, text_ids):
        text_length = len(text_ids)
        input_ids, output_ids = [], []
        ratio = np.random.choice([0.45, 0.35, 0.25, 0.15], p=[0.2, 0.4, 0.3, 0.1])  # 随机选取mask比例
        replaced_probability_list = bernoulli.rvs(p=ratio, size=text_length)  # 元素为0/1的列表
        replaced_list = bernoulli.rvs(p=0, size=text_length)  # 1表示需要mask，0表示不需要mask, 初始化时全0
        idx = 0
        while idx < text_length:
            if (replaced_probability_list[idx] == 1) and text_ids[idx] not in self.special_token:
                ngram = np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.1, 0.1])  # 若要mask，进行x_gram mask的概率
                L = idx
                R = idx + ngram
                # 1 表示用 【MASK】符号进行mask，2表示随机替换，3表示不进行替换
                mask_partten = np.random.choice([1, 2, 3], p=[0.8, 0.1, 0.1])
                replaced_list[L: R] = mask_partten
                idx = R
                if idx < text_length:
                    replaced_probability_list[R] = 0  # 防止连续mask
            idx += 1

        for r, i in zip(replaced_list, text_ids):
            if r == 1:
                # 使用【mask】替换
                input_ids.append(self.mask_token)
                output_ids.append(i)  # mask预测自己
            elif r == 2:
                # 随机的一个词预测自己，随机词从训练集词表中取，有小概率抽到自己
                input_ids.append(random.choice(text_ids))
                output_ids.append(i)
            elif r == 3:
                # 不进行替换
                input_ids.append(i)
                output_ids.append(i)
            else:
                # 没有被选中要被mask
                input_ids.append(i)
                output_ids.append(-100)

        return input_ids, output_ids

    def encode(self, item):
        text = item
        text_ids = self.tokenizer.encode(text, truncation=True, max_length=512)[1:-1]

        input_ids, labels = self.ngram_mask(text_ids)  # ngram mask
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        labels = [-100] + labels + [-100]

        attention_mask = [1] * len(input_ids)

        return input_ids, attention_mask, labels

    @staticmethod
    def collate(batch):
        batch_input_ids, batch_attention_mask = [], []
        batch_labels = []

        for item in batch:
            input_ids, attention_mask, label = item
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(label)

        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids)).long()
        batch_attention_mask = torch.tensor(sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(sequence_padding(batch_labels, padding=-100)).long()

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels
        }


def build_optimizer_and_scheduler(model, num_total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    param_optimizer = []

    for name, param in model.named_parameters():
        param_optimizer.append((name, param))

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': 2e-5},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': 2e-5},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-6)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.05 * num_total_steps, num_training_steps=num_total_steps * 1.05
    )

    return optimizer, scheduler


def train():
    with open('./OAG_data/DBLP-Citation-network-V15.1.json', 'r') as f:
        data = []
        for line in f.readlines():
            data.append(json.loads(line))
    clear_data = []


    for item in tqdm(data):
        clear_data.append(
            {'id': item['id'], 'title': item['title'], 'abstract': item['abstract'], 'keywords': item['keywords'],
             'year': item['year'], 'authors': item['authors'], 'venue': item['venue']})
    clear_data = pd.DataFrame(clear_data)

    print('加载数据完成！')
    dataset = BertDataSet_MLM(MODEL_PATH, clear_data)
    sampler = DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate, sampler=sampler,num_workers=0)

    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)

    num_total_steps = len(train_dataloader)
    optimizer, scheduler = build_optimizer_and_scheduler(model, num_total_steps)
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    best_score = 0
    step = 0
    model.train()
    sampler.set_epoch(1)
    with tqdm(train_dataloader) as pBar:
        pBar.set_description(desc=f'[training]')
        for batch in pBar:
            step += 1
            with autocast():
                for k in batch.keys():
                    batch[k] = batch[k].to(device)
                output = model(**batch)
                loss = output.loss
                loss = loss.mean()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            optimizer.zero_grad()
            scheduler.step()
            scaler.update()
            pBar.set_postfix(loss=loss.item())
            if (step + 1) % 5000 == 0 and rank == 0:
                model.eval()
                torch.save(model.module.state_dict(), f'./save_pretrain/model_save_deberta_large/pretrain_step{step+1}.bin')
                model.train()


if __name__ == '__main__':

    ###This code is used to pretrain bert-like model, pretrain data is DBLP.json, please replace the data path or model path into your path


    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    setup_seed(seed=42)
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    warnings.filterwarnings('ignore')
    os.makedirs('./save_pretrain/model_save_deberta_large', exist_ok=True)
    MODEL_PATH = 'microsoft/deberta-v3-large'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    train()
