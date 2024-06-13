import torch
import numpy as np
import os
import random
seed=42
print("Seed: {}".format(seed))
from fuzzywuzzy import fuzz
import pickle as pkl
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
from os.path import join
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AutoConfig
from transformers.optimization import AdamW
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, Dataset
from tqdm import trange
from sklearn.metrics import classification_report, precision_recall_fscore_support, average_precision_score
import logging
import json
import pandas as pd
import torch.nn as nn
import utils
import settings
import warnings
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH = 2048


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
        if len(input_ids) > (self.max_seq_len - 3):
            input_ids = input_ids[:(self.max_seq_len - 3)]
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

class KddMoveLearningDataSet(Dataset):
    def __init__(self, text, labels, tokenizer, max_seq_len, embed_npy):
        self.text = text
        self.labels = labels
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.embed_npy = np.load(embed_npy)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        label = self.labels[item]
        text = text
        input_ids = self.tokenizer.encode(text,add_special_tokens=False)
        if len(input_ids) > (self.max_seq_len - 3):
            input_ids = input_ids[:(self.max_seq_len - 3)]
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        return input_ids, label, self.embed_npy[item]

    @staticmethod
    def collate(batch):
        batch_input_ids, batch_labels = [], []
        batch_embed = []
        for input_ids, label, embed in batch:
            batch_input_ids.append(input_ids)
            batch_labels.append(label)
            batch_embed.append(embed)
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
        batch_embed = np.concatenate(batch_embed, axis=0)
        batch_embed = torch.from_numpy(batch_embed)
        return {'input_ids':batch_input_ids, 'attention_mask':batch_attention_mask, 'labels':batch_labels, 'input_embed':batch_embed}

from functools import partial
def get_data_loader(dataset, batch_size, shuffle=True):
    dataloader_class = partial(DataLoader)
    dataloader = dataloader_class(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate, num_workers=8)
    return dataloader


from sklearn.metrics import average_precision_score


def evaluate(model, dataloader, device, criterion, dev_idx):
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    for step, batch in tqdm(enumerate(dataloader),total=len(dataloader)):
        for k, v in batch.items():
            batch[k] = v.cuda()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                input_embed = model.module.get_input_embeddings()(batch['input_ids'])
                attention_mask = torch.concat([torch.tensor([[1] for i in range(batch['attention_mask'].shape[0])],
                                                            device=batch['attention_mask'].device),
                                               batch['attention_mask']], dim=1)
                global_attention_mask = torch.zeros_like(attention_mask)
                global_attention_mask[:, 0] = 1
                input_embed = torch.cat([batch['input_embed'].view(input_embed.shape[0], 1, 1024), input_embed], dim = 1)
                outputs = model(inputs_embeds=input_embed, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
                logits = outputs.logits
        outputs = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]
        label_ids = batch['labels'].to('cpu').numpy()

        predicted_labels += list(outputs)
        correct_labels += list(label_ids)
        nb_eval_steps += 1


    dev = pd.DataFrame({'pred':predicted_labels, 'label':correct_labels, 'idx':dev_idx})
    dev_idx_s = set(dev_idx)

    map = []
    for idx in dev_idx_s:
        cut = dev[dev['idx'] == idx].reset_index(drop=True)
        p = cut['pred'].tolist()
        l = cut['label'].tolist()
        ap = average_precision_score(l, p)
        map.append(ap)
    map = np.mean(map)


    return map, correct_labels, predicted_labels


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

def train(year=2023, model_name="scibert", fold=0):
    print("model name", model_name)
    train_texts = []
    dev_texts = []
    train_labels = []
    dev_labels = []
    train_idx = []
    dev_idx = []
    data_year_dir = join(settings.DATA_TRACE_DIR, "PST")
    print("data_year_dir", data_year_dir)
    train = pd.read_pickle(f'./data/train_extral_fold{fold}.pickle')
    valid = pd.read_pickle(f'./data/valid_extral_fold{fold}.pickle')
    print(f"Fold:{fold}")
    for _, row in train.iterrows():
        train_texts.append(row['text'])
        train_labels.append(row['label'])
        train_idx.append(row['idx'])

    for _, row in valid.iterrows():
        dev_texts.append(row['text'])
        dev_labels.append(row['label'])
        dev_idx.append(row['idx'])
    print(train_texts[0])
    print("Train size:", len(train_texts))
    print("Dev size:", len(dev_texts))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weight = len(train_labels) / (2 * np.bincount(train_labels))
    class_weight = torch.Tensor(class_weight).to(device)
    print("Class weight:", class_weight)

    BERT_MODEL = "microsoft/deberta-v3-large"
    config = AutoConfig.from_pretrained(BERT_MODEL)
    config.update({'max_position_embeddings': MAX_SEQ_LENGTH, 'num_labels': 2})

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, config=config)
    state_dict = torch.load('./save_pretrain/model_save_deberta_large/pretrain_step50000.bin', map_location='cpu')
    model.load_state_dict(state_dict,strict=False)
    model = torch.nn.parallel.DataParallel(model)
    model.to(device)


    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    # criterion = torch.nn.CrossEntropyLoss()

    train_dataset = KddMoveLearningDataSet(train_texts, train_labels, tokenizer, MAX_SEQ_LENGTH, f'./data/train_grafting_learing_hidden_state_pretrain_fold{fold}.npy')
    dev_dataset = KddMoveLearningDataSet(dev_texts, dev_labels, tokenizer, MAX_SEQ_LENGTH, f'./data/dev_grafting_learing_hidden_state_pretrain_fold{fold}.npy')


    BATCH_SIZE = 16
    train_dataloader = get_data_loader(train_dataset, BATCH_SIZE, shuffle=True)
    dev_dataloader = get_data_loader(dev_dataset, 32, shuffle=False)

    GRADIENT_ACCUMULATION_STEPS = 1
    NUM_TRAIN_EPOCHS = 20
    LEARNING_RATE = 1e-5
    WARMUP_PROPORTION = 0.1
    MAX_GRAD_NORM = 5

    num_train_steps = int(len(train_dataloader.dataset) / BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(WARMUP_PROPORTION * num_train_steps)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_steps)

    OUTPUT_DIR = join(settings.OUT_DIR, model_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_FILE_NAME = "pytorch_model.bin"
    PATIENCE = 5
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    ema = EMA(model, 0.95)
    ema.register()
    # fgm = FGM(model, emb_name='word_embeddings', epsilon=0.25)
    best_map = 0
    for epoch in range(int(NUM_TRAIN_EPOCHS)):
        model.train()
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
            for k, v in batch.items():
                batch[k] = v.cuda()
            with autocast():
                input_embed = model.module.get_input_embeddings()(batch['input_ids'])
                attention_mask = torch.concat([torch.tensor([[1] for i in range(batch['attention_mask'].shape[0])], device=batch['attention_mask'].device), batch['attention_mask']], dim=1)
                input_embed = torch.cat([batch['input_embed'].view(input_embed.shape[0], 1, 1024), input_embed], dim = 1)
                global_attention_mask = torch.zeros_like(attention_mask)
                global_attention_mask[:, 0] = 1
                logits1 = model(inputs_embeds = input_embed, attention_mask=attention_mask, global_attention_mask=global_attention_mask).logits
                logits2 = model(inputs_embeds = input_embed, attention_mask=attention_mask, global_attention_mask=global_attention_mask).logits
                loss = (criterion(logits1, batch['labels']) + criterion(logits2, batch['labels'])) / 2 + compute_kl_loss(logits1, logits2)
            scaler.scale(loss).backward()


            tr_loss += loss.item()

            scaler.step(optimizer)
            optimizer.zero_grad()
            scheduler.step()
            scaler.update()
            if ema:
                ema.update()
        if ema:
            ema.apply_shadow()

        map, _, _ = evaluate(model, dev_dataloader, device, criterion, dev_idx)

        print("Dev Map:", map)
        model_to_save = model.module if hasattr(model, 'module') else model
        if map > best_map:
            best_map = map
            output_model_file = os.path.join(f'./out/deberta_large_3fold_{fold}', f'pytorch_model.bin')
            torch.save(model_to_save.state_dict(), output_model_file)
        if ema:
            ema.restore()

def eval_test_papers_bert(model_name="scibert"):
    for fold in range(3):
        print("model name", model_name)
        train_texts = []
        dev_texts = []
        train_labels = []
        dev_labels = []
        train_idx = []
        dev_idx = []
        data_year_dir = join(settings.DATA_TRACE_DIR, "PST")
        print("data_year_dir", data_year_dir)

        train = pd.read_pickle(f'./data/train_extral_fold{fold}.pickle')
        valid = pd.read_pickle(f'./data/valid_extral_fold{fold}.pickle')

        for _, row in train.iterrows():
            train_texts.append(row['text'])
            train_labels.append(row['label'])
            train_idx.append(row['idx'])

        for _, row in valid.iterrows():
            dev_texts.append(row['text'])
            dev_labels.append(row['label'])
            dev_idx.append(row['idx'])

        print("Train size:", len(train_texts))
        print("Dev size:", len(dev_texts))
        print("Train size:", len(train_labels))
        print("Dev size:", len(dev_labels))


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        BERT_MODEL = "microsoft/deberta-v3-large"
        config = AutoConfig.from_pretrained(BERT_MODEL)
        config.update({'max_position_embeddings': MAX_SEQ_LENGTH, 'num_labels': 2})

        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, config=config)
        state_dict = torch.load('./out/grafting_learning_deberta_large/pretrain_epoch1.bin', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model = model.deberta
        model = torch.nn.parallel.DataParallel(model)
        model.to(device)
        mean_pool = MeanPooling()
        mean_pool = torch.nn.parallel.DataParallel(mean_pool)
        mean_pool.to(device)

        train_dataset = KddDataSet(train_texts, train_labels, tokenizer, MAX_SEQ_LENGTH)
        dev_dataset = KddDataSet(dev_texts, dev_labels, tokenizer, MAX_SEQ_LENGTH)


        BATCH_SIZE = 64
        train_dataloader = get_data_loader(train_dataset, BATCH_SIZE, shuffle=False)
        dev_dataloader = get_data_loader(dev_dataset, BATCH_SIZE, shuffle=False)
        model.eval()

        total = []
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            for k, v in batch.items():
                batch[k] = v.cuda()

            with torch.no_grad():
                r = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).last_hidden_state
                r = mean_pool(r, batch['attention_mask']).detach().view(batch['input_ids'].shape[0], 1, 1024).cpu().numpy()
                total.append(r)
        final = np.concatenate(total, axis=0)
        np.save(f'./data/train_grafting_learing_hidden_state_pretrain_fold{fold}.npy', final)

        total = []
        for step, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
            for k, v in batch.items():
                batch[k] = v.cuda()

            with torch.no_grad():
                r = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).last_hidden_state
                r = mean_pool(r, batch['attention_mask']).detach().view(batch['input_ids'].shape[0], 1,
                                                                        1024).cpu().numpy()
                total.append(r)
        final = np.concatenate(total, axis=0)
        np.save(f'./data/dev_grafting_learing_hidden_state_pretrain_fold{fold}.npy', final)

def get_test_embed(model_name='scibert'):
    print("model name", model_name)
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")

    BERT_MODEL = "microsoft/deberta-v3-large"
    config = AutoConfig.from_pretrained(BERT_MODEL)
    config.update({'max_position_embeddings': MAX_SEQ_LENGTH, 'num_labels': 2})
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    os.makedirs('./data/test_embed', exist_ok=True)
    sub_example_dict = utils.load_json(data_dir, "submission_example_valid.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, config=config)
    model.load_state_dict(torch.load('./out/grafting_learning_deberta_large/pretrain_epoch1.bin', map_location='cpu'))
    model = model.deberta
    model = torch.nn.parallel.DataParallel(model)
    mean_pool = MeanPooling()
    mean_pool = torch.nn.parallel.DataParallel(mean_pool)
    mean_pool.to(device)
    model.to(device)
    model.eval()

    BATCH_SIZE = 64
    # metrics = []
    # f_idx = 0

    xml_dir = join(data_dir, "paper-xml")
    sub_dict = {}
    save_paper = {}
    total = 0
    miss = 0
    for paper in tqdm(papers):
        cur_pid = paper["_id"]
        file = join(xml_dir, cur_pid + ".xml")
        f = open(file, encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()

        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            b_idx = int(bid[1:]) + 1
            if b_idx > len(sub_example_dict[cur_pid]):
                continue
            title = ""
            flag = False
            if ref.analytic is not None and ref.analytic.title is not None:
                title += ref.analytic.title.text.lower()
                flag = True
            if ref.monogr is not None and ref.monogr.title is not None and flag is False:
                title += ref.monogr.title.text.lower()
            bid_to_title[bid] = title
            if b_idx > n_refs:
                n_refs = b_idx
        total += n_refs
        miss += (n_refs - len(bid_to_title.keys()))
        bib_to_contexts = utils.find_bib_context(xml)
        # bib_sorted = sorted(bib_to_contexts.keys())
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]

        y_score = [0] * n_refs

        assert len(sub_example_dict[cur_pid]) == n_refs

        contexts_sorted = [" ".join(bib_to_contexts[bib]) for bib in bib_sorted]
        test_features = KddDataSet(contexts_sorted, y_score, tokenizer, MAX_SEQ_LENGTH)
        test_dataloader = get_data_loader(test_features, BATCH_SIZE, shuffle=False)

        total = []
        for step, batch in enumerate(test_dataloader):
            for k, v in batch.items():
                batch[k] = v.cuda()

            with torch.no_grad():
                r = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).last_hidden_state
                r = mean_pool(r, batch['attention_mask']).detach().view(batch['input_ids'].shape[0], 1, 1024).cpu().numpy()
                total.append(r)
        final = np.concatenate(total, axis=0)
        np.save(f'./data/test_embed/{cur_pid}.npy', final)


def eval(model_name="scibert", fold=0):
    print("model name", model_name)
    train_texts = []
    dev_texts = []
    train_labels = []
    dev_labels = []
    train_idx = []
    dev_idx = []
    data_year_dir = join(settings.DATA_TRACE_DIR, "PST")
    print("data_year_dir", data_year_dir)


    train = pd.read_pickle(f'./data/train_extral_fold{fold}.pickle')
    valid = pd.read_pickle(f'./data/valid_extral_fold{fold}.pickle')

    for _, row in train.iterrows():
        train_texts.append(row['text'])
        train_labels.append(row['label'])
        train_idx.append(row['idx'])

    for _, row in valid.iterrows():
        dev_texts.append(row['text'])
        dev_labels.append(row['label'])
        dev_idx.append(row['idx'])

    print("Train size:", len(train_texts))
    print("Dev size:", len(dev_texts))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    BERT_MODEL = "microsoft/deberta-v3-large"
    config = AutoConfig.from_pretrained(BERT_MODEL)
    config.update({'max_position_embeddings': MAX_SEQ_LENGTH, 'num_labels': 2})

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, config=config)
    state_dict = torch.load(f'./out/deberta_large_3fold_{fold}/pytorch_model.bin', map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model = torch.nn.parallel.DataParallel(model)
    model.to(device)

    dev_dataset = KddMoveLearningDataSet(dev_texts, dev_labels, tokenizer, MAX_SEQ_LENGTH, f'./data/dev_grafting_learing_hidden_state_pretrain_fold{fold}.npy')

    BATCH_SIZE = 64
    dev_dataloader = get_data_loader(dev_dataset, BATCH_SIZE, shuffle=False)
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []
    autocast = torch.cuda.amp.autocast
    for step, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
        for k, v in batch.items():
            batch[k] = v.cuda()

        with torch.no_grad():
            with autocast():
                input_embed = model.module.get_input_embeddings()(batch['input_ids'])
                attention_mask = torch.concat([torch.tensor([[1] for i in range(batch['attention_mask'].shape[0])],
                                                            device=batch['attention_mask'].device),
                                               batch['attention_mask']], dim=1)
                input_embed = torch.cat([batch['input_embed'].view(input_embed.shape[0], 1, 1024), input_embed], dim=1)
                outputs = model(inputs_embeds=input_embed, attention_mask=attention_mask)
                logits = outputs.logits
        outputs = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]
        label_ids = batch['labels'].to('cpu').numpy()

        predicted_labels += list(outputs)
        correct_labels += list(label_ids)
    dev = pd.DataFrame({'pred':predicted_labels, 'label':correct_labels, 'idx':dev_idx})
    dev_idx_s = set(dev_idx)

    map = []
    for idx in dev_idx_s:
        cut = dev[dev['idx'] == idx].reset_index(drop=True)
        p = cut['pred'].tolist()
        l = cut['label'].tolist()
        ap = average_precision_score(l, p)
        map.append(ap)
    map = np.mean(map)
    print(map)
    valid['pred'] = predicted_labels
    valid.to_pickle(f'./data/valid_extral_withlogits_fold{fold}.pickle')


import pickle
def gen_kddcup_valid_submission_bert(model_name="scibert"):
    print("model name", model_name)
    data_dir = join('./data/', "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")

    BERT_MODEL = "microsoft/deberta-v3-large"
    config = AutoConfig.from_pretrained(BERT_MODEL)
    config.update({'max_position_embeddings': MAX_SEQ_LENGTH, 'num_labels': 2})
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

    sub_example_dict = utils.load_json(data_dir, "submission_example_test.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, config=config)
    model.load_state_dict(torch.load(f"./out/deberta_large_3fold_2/pytorch_model.bin"))
    model = torch.nn.parallel.DataParallel(model)

    model.to(device)
    model.eval()

    BATCH_SIZE = 64
    # metrics = []
    # f_idx = 0
    autocast = torch.cuda.amp.autocast

    xml_dir = join(data_dir, "paper-xml")
    sub_dict = {}
    total = 0
    miss = 0
    for paper in tqdm(papers):
        cur_pid = paper["_id"]
        file = join(xml_dir, cur_pid + ".xml")
        f = open(file, encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()

        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            b_idx = int(bid[1:]) + 1
            if b_idx > len(sub_example_dict[cur_pid]):
                continue
            title = ""
            flag = False
            if ref.analytic is not None and ref.analytic.title is not None:
                title += ref.analytic.title.text.lower()
                flag = True
            if ref.monogr is not None and ref.monogr.title is not None and flag is False:
                title += ref.monogr.title.text.lower()
            bid_to_title[bid] = title
            if b_idx > n_refs:
                n_refs = b_idx
        total += n_refs
        miss += (n_refs - len(bid_to_title.keys()))
        bib_to_contexts = utils.find_bib_context(xml)
        # bib_sorted = sorted(bib_to_contexts.keys())
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]

        y_score = [0] * n_refs

        assert len(sub_example_dict[cur_pid]) == n_refs

        contexts_sorted = [" ".join(bib_to_contexts[bib]) for bib in bib_sorted]
        test_features = KddMoveLearningDataSet(contexts_sorted, y_score, tokenizer, MAX_SEQ_LENGTH, f'./data/test_embed/{cur_pid}.npy')
        test_dataloader = get_data_loader(test_features, BATCH_SIZE, shuffle=False)

        predicted_scores = []
        for step, batch in enumerate(test_dataloader):
            for k, v in batch.items():
                batch[k] = v.cuda()

            with torch.no_grad():
                input_embed = model.module.get_input_embeddings()(batch['input_ids'])
                attention_mask = torch.concat([torch.tensor([[1] for i in range(batch['attention_mask'].shape[0])],
                                                            device=batch['attention_mask'].device),
                                               batch['attention_mask']], dim=1)
                input_embed = torch.cat([batch['input_embed'].view(input_embed.shape[0], 1, 1024), input_embed], dim=1)
                with autocast():
                    outputs = model(inputs_embeds=input_embed, attention_mask=attention_mask)
                    logits = torch.softmax(outputs.logits, dim=1)

            cur_pred_scores = logits[:, 1].to('cpu').numpy()
            predicted_scores.extend(list(cur_pred_scores))

        for ii in range(len(predicted_scores)):
            bib_idx = int(bib_sorted[ii][1:])
            # print("bib_idx", bib_idx)
            y_score[bib_idx] = float(predicted_scores[ii])

        sub_dict[cur_pid] = y_score
    print(f"Total Ref:{total}, Miss Ref:{miss}")
    with open('./data/submission.json', 'w') as f:
        json.dump(sub_dict, f, indent=4, ensure_ascii=False)




if __name__ == "__main__":
    origin_dir = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/chenhaoru/zhipu_kdd/track2/'
    settings.OUT_DIR = origin_dir + 'out/deberta_large_3fold'

    eval_test_papers_bert(model_name="deberta-large")   #利用嫁接模型得到每个样本的hidden_state
    for fold in range(3):
        train(model_name="deberta-large", fold=fold)  # 训练
        eval(model_name='deberta-large', fold=fold)
    get_test_embed(model_name="deberta-large")   #拿到测试集的嫁接模型的hidden_state
    gen_kddcup_valid_submission_bert(model_name="deberta-large")   #推理
