import os
from os.path import join
from tqdm import tqdm
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import trange
from sklearn.metrics import classification_report, precision_recall_fscore_support, average_precision_score
import logging
import faiss
import utils
import settings

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH = 2048

import json


def prepare_bert_input():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    idx_train = []
    idx_valid = []
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    n_papers = len(papers)
    papers = sorted(papers, key=lambda x: x["_id"])
    n_train = int(n_papers * 2 / 3)
    # n_valid = n_papers - n_train

    papers_train = papers[:n_train]
    papers_valid = papers[n_train:]

    pids_train = {p["_id"] for p in papers_train}
    pids_valid = {p["_id"] for p in papers_valid}

    in_dir = join(data_dir, "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        if f.endswith(".xml"):
            files.append(f)

    pid_to_source_titles = dd(list)
    for paper in tqdm(papers):
        pid = paper["_id"]
        for ref in paper["refs_trace"]:
            pid_to_source_titles[pid].append(ref["title"].lower())

    # files = sorted(files)
    # for file in tqdm(files):
    total = []
    cnt_train, cnt_valid = 0, 0
    for cur_pid in tqdm(pids_train):
        # cur_pid = file.split(".")[0]
        # if cur_pid not in pids_train and cur_pid not in pids_valid:
        # continue
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
        cur_authors_paper = set()
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

        n_pos = len(cur_pos_bib)
        n_neg = n_pos * 10
        cur_neg_bib_sample = np.random.choice(list(cur_neg_bib), n_neg, replace=True)

        cur_x = x_train
        cur_y = y_train
        cur_idx = idx_train

        for bib in cur_pos_bib:
            cur_context = " ".join(bib_to_contexts[bib])
            cur_x.append(cur_context)
            cur_y.append(1)
            cur_idx.append(cur_pid)

        for bib in cur_neg_bib_sample:
            cur_context = " ".join(bib_to_contexts[bib])
            cur_x.append(cur_context)
            cur_y.append(0)
            cur_idx.append(cur_pid)

    print(f"Train miss label:{np.sum(total)}, mean:{np.mean(total)}")
    print(f"Train total pos label:{np.sum(y_train)}")
    print(f"Train miss author's paper:{cnt_train}")
    total = []
    for cur_pid in tqdm(pids_valid):
        # cur_pid = file.split(".")[0]
        # if cur_pid not in pids_train and cur_pid not in pids_valid:
        # continue
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
        total.append(len(source_titles) - len(cur_pos_bib))

        cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib

        if not flag:
            continue

        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue

        bib_to_contexts = utils.find_bib_context(xml)

        n_pos = len(cur_pos_bib)

        cur_x = x_valid
        cur_y = y_valid
        cur_idx = idx_valid

        for bib in cur_pos_bib:
            cur_context = " ".join(bib_to_contexts[bib])
            cur_x.append(cur_context)
            cur_y.append(1)
            cur_idx.append(cur_pid)

        for bib in list(cur_neg_bib):
            cur_context = " ".join(bib_to_contexts[bib])
            cur_x.append(cur_context)
            cur_y.append(0)
            cur_idx.append(cur_pid)
    print(f"Val miss label:{np.sum(total)}, mean:{np.mean(total)}")
    print(f"Val total pos label:{np.sum(y_valid)}")
    print(f"Val miss author's paper:{cnt_valid}")

    print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid))

    with open(join(data_dir, "more_info_bib_context_train.txt"), "w", encoding="utf-8") as f:
        for line in x_train:
            f.write(line + "\n")

    with open(join(data_dir, "more_info_bib_context_valid.txt"), "w", encoding="utf-8") as f:
        for line in x_valid:
            f.write(line + "\n")

    with open(join(data_dir, "more_info_bib_context_train_label.txt"), "w", encoding="utf-8") as f:
        for line in y_train:
            f.write(str(line) + "\n")

    with open(join(data_dir, "more_info_bib_context_valid_label.txt"), "w", encoding="utf-8") as f:
        for line in y_valid:
            f.write(str(line) + "\n")

    with open(join(data_dir, "more_info_bib_context_train_idx.txt"), "w", encoding="utf-8") as f:
        for line in idx_train:
            f.write(str(line) + "\n")

    with open(join(data_dir, "more_info_bib_context_valid_idx.txt"), "w", encoding="utf-8") as f:
        for line in idx_valid:
            f.write(str(line) + "\n")


def prepare_llm_input():
    x_train = []
    y_train = []
    idx_train = []
    abstract_train = []
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    papers = sorted(papers, key=lambda x: x["_id"])
    bert_pred = pd.read_pickle('./data/valid_extral_withlogits_all_addcite.pickle')
    bert_pred['rank'] = bert_pred.groupby('idx')['pred'].rank(ascending=False)
    bert_pred['rank'] = bert_pred['rank'].apply(lambda x: int(x))
    papers_train = papers

    pids_train = {p["_id"] for p in papers_train}

    in_dir = join(data_dir, "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        if f.endswith(".xml"):
            files.append(f)

    pid_to_source_titles = dd(list)
    for paper in tqdm(papers):
        pid = paper["_id"]
        for ref in paper["refs_trace"]:
            pid_to_source_titles[pid].append(ref["title"].lower())

    # files = sorted(files)
    # for file in tqdm(files):
    total = []
    all_source_titles, all_ref_titles = [], []
    cnt_train, cnt_valid = 0, 0
    moremoremore = []
    ref_and_true_label = []
    for cur_pid in tqdm(pids_train):
        # cur_pid = file.split(".")[0]
        # if cur_pid not in pids_train and cur_pid not in pids_valid:
        # continue
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
                    ref_and_true_label.append({'ref': cur_ref_title, 'label': label_title})

        cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib
        total.append(len(source_titles) - len(cur_pos_bib))
        if not flag:
            continue

        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue

        n_pos = len(cur_pos_bib)
        n_neg = min(n_pos * 10, len(cur_neg_bib))
        cur_neg_bib_sample = np.random.choice(list(cur_neg_bib), n_neg, replace=True)

        title = bs.find('titleStmt').find('title').text
        ture_source_title = title
        text = "Title:\n"
        text += title + '\n'
        # 提取摘要
        abstract = bs.find('abstract')
        if abstract:
            text += "Abstract:\n"
            for p in abstract.find_all('p'):
                text += p.text + '\n'

        body = bs.find('body')
        text += 'Body:\n'
        if body:
            for div in body.find_all('div'):
                texts = []
                for tag in div.descendants:
                    if tag.name:
                        if tag.attrs and 'type' in tag.attrs.keys() and tag.attrs['type'] == 'bibr':
                            try:
                                texts.append(tag.attrs['target'] + ' ')
                            except Exception:
                                a = 1
                        texts.append(tag.text)
                text += ''.join(texts) + '\n'

        references = bs.find('listBibl').find_all('biblStruct')
        text += 'References:\n'
        cut = bert_pred[bert_pred['idx'] == cur_pid].reset_index(drop=True)

        # 按照文档中定义的顺序（即XML中的出现顺序）遍历参考文献
        for i, ref in enumerate(references):
            # 提取作者信息
            authors = [author.text for author in ref.find_all('persName')]
            author_str = ' '.join(authors)

            # 提取标题信息
            title = ref.find('title').text if ref.find('title') else 'No title'

            rank = cut[cut['bib'] == f"b{i}"].reset_index(drop=True)
            text += f"{i}, {title}, {author_str}.\n"

        for bib in cur_pos_bib:
            bid_title = bid_to_title[bib]
            ranks = cut[cut['bib'] == bib].reset_index(drop=True)
            assert len(ranks) == 1
            rank = ranks['rank'][0]
            all_source_titles.append(ture_source_title)
            all_ref_titles.append(bid_title)
            tmp = {}
            for k in ['occur_cnt']:
                tmp[k] = ranks[k][0]
            tmp['ref_bib'] = bib
            tmp['ref_rank'] = rank
            cur_context = "You are a computer thesis expert, and you have to judge whether a reference is the source paper of the original text based on the original text of the paper."
            cur_context += """The following points define whether a reference is a source paper:\nIs the main idea of paper p inspired by the reference？\nIs the core method of paper p derived from the reference？\nIs the reference essential for paper p? Without the work of this reference, paper p cannot be completed.\n"""
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}.\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}.\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}.\n"
            cur_context += text
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}.\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}.\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}.\n"
            x_train.append(cur_context)
            y_train.append(1)
            idx_train.append(cur_pid)
            moremoremore.append(tmp)

        for bib in cur_neg_bib:
            bid_title = bid_to_title[bib]
            ranks = cut[cut['bib'] == bib].reset_index(drop=True)
            assert len(ranks) == 1
            rank = ranks['rank'][0]
            all_source_titles.append(ture_source_title)
            all_ref_titles.append(bid_title)
            tmp = {}
            for k in ['occur_cnt']:
                tmp[k] = ranks[k][0]
            tmp['ref_bib'] = bib
            tmp['ref_rank'] = rank
            cur_context = "You are a computer thesis expert, and you have to judge whether a reference is the source paper of the original text based on the original text of the paper."
            cur_context += """The following points define whether a reference is a source paper:\nIs the main idea of paper p inspired by the reference？\nIs the core method of paper p derived from the reference？\nIs the reference essential for paper p? Without the work of this reference, paper p cannot be completed.\n"""
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}.\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}.\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}.\n"
            cur_context += text
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}.\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}.\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}.\n"
            x_train.append(cur_context)
            y_train.append(0)
            idx_train.append(cur_pid)
            moremoremore.append(tmp)

    more_info = pd.DataFrame({'source_title': all_source_titles, 'ref_title': all_ref_titles})
    moremore = pd.DataFrame(moremoremore)
    train = pd.DataFrame({'text': x_train, 'label': y_train, 'idx': idx_train})
    train = pd.concat([train, more_info, moremore], axis=1)
    # train.to_pickle('./data/llm_train.pickle')
    # valid.to_pickle('./data/llm_valid.pickle')
    train.to_pickle('./data/llm_notsample_with_allinfo_addbib_addcite.pickle')
    # ref_and_true_label = pd.DataFrame(ref_and_true_label)
    # ref_and_true_label.to_pickle('./data/ref_and_true_label.pickle')


def prepare_llm_fakelabel_input():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    idx_train = []
    idx_valid = []
    abstract_train = []
    abstract_valid = []
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    fakepapers = utils.load_json(data_dir, "paper_source_gen_by_rule.json")
    bert_pred = pd.read_pickle('./data/relu_data_all_data_with_predlogits_with_moreinfo.pickle')
    bert_pred['rank'] = bert_pred.groupby('idx')['pred'].rank(ascending=False)
    bert_pred['rank'] = bert_pred['rank'].apply(lambda x: int(x))
    papers = []
    for k, v in fakepapers.items():
        papers.append({'_id': k, 'refs_trace': v})
    papers_train = papers

    pids_train = {p["_id"] for p in papers_train}

    in_dir = join(data_dir, "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        if f.endswith(".xml"):
            files.append(f)

    pid_to_source_titles = dd(list)
    for paper in tqdm(papers):
        pid = paper["_id"]
        for k, v in paper["refs_trace"].items():
            pid_to_source_titles[pid].append(v.lower())

    # files = sorted(files)
    # for file in tqdm(files):
    total = []
    cnt_train, cnt_valid = 0, 0
    ref_and_true_label = []
    more_info = []
    for cur_pid in tqdm(pids_train):
        # cur_pid = file.split(".")[0]
        # if cur_pid not in pids_train and cur_pid not in pids_valid:
        # continue
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
                    ref_and_true_label.append({'ref': cur_ref_title, 'label': label_title})

        cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib
        total.append(len(source_titles) - len(cur_pos_bib))
        if not flag:
            continue

        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue

        n_pos = len(cur_pos_bib)
        n_neg = min(n_pos * 10, len(cur_neg_bib))

        title = bs.find('titleStmt').find('title').text
        text = "Title:\n"
        text += title + '\n'
        # 提取摘要
        abstract = bs.find('abstract')
        if abstract:
            text += "Abstract:\n"
            for p in abstract.find_all('p'):
                text += p.text + '\n'

        # 提取正文
        body = bs.find('body')
        text += 'Body:\n'
        if body:
            for div in body.find_all('div'):
                texts = []
                for tag in div.descendants:
                    if tag.name:
                        texts.append(tag.text)
                text += ''.join(texts) + '\n'

        references = bs.find('listBibl').find_all('biblStruct')
        text += 'References:\n'
        # 按照文档中定义的顺序（即XML中的出现顺序）遍历参考文献
        for i, ref in enumerate(references):
            # 提取作者信息
            authors = [author.text for author in ref.find_all('persName')]
            author_str = ' '.join(authors)

            # 提取标题信息
            title = ref.find('title').text if ref.find('title') else 'No title'

            text += f"{i}, {title}, {author_str}.\n"
        cut = bert_pred[bert_pred['idx'] == cur_pid].reset_index(drop=True)
        for bib in cur_pos_bib:
            bid_title = bid_to_title[bib]
            ranks = cut[cut['bib'] == bib].reset_index(drop=True)
            assert len(ranks) == 1
            rank = ranks['rank'][0]
            tmp = {}
            for k in ['source_year', 'source_venue', 'source_keywords', 'source_cite', 'ref_title', 'ref_abstract',
                      'ref_year', 'ref_venue', 'ref_keywords', 'ref_authors',
                      'ref_cite', 'occur_cnt']:
                tmp[k] = ranks[k][0]
            cur_context = "You are a computer thesis expert, and you have to judge whether a reference is the source paper of the original text based on the original text of the paper."
            cur_context += """The following points define whether a reference is a source paper:\nIs the main idea of paper p inspired by the reference？\nIs the core method of paper p derived from the reference？\nIs the reference essential for paper p? Without the work of this reference, paper p cannot be completed.\n"""
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += text
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            x_train.append(cur_context)
            y_train.append(1)
            idx_train.append(cur_pid)
            more_info.append(tmp)

    train = pd.DataFrame({'text': x_train, 'label': y_train, 'idx': idx_train})
    more_info = pd.DataFrame(more_info)
    train = pd.concat([train, more_info], axis=1)
    print(train.shape)
    # train.to_pickle('./data/llm_train.pickle')
    # valid.to_pickle('./data/llm_valid.pickle')
    train.to_pickle('./data/llm_fakelabel_all_pos_withmoreinfo.pickle')
    # ref_and_true_label = pd.DataFrame(ref_and_true_label)
    # ref_and_true_label.to_pickle('./data/ref_and_true_label.pickle')


def prepare_llm_sentence_transformer_input():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    idx_train = []
    idx_valid = []
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    n_papers = len(papers)
    papers = sorted(papers, key=lambda x: x["_id"])
    n_train = int(n_papers * 3 / 4)
    # n_valid = n_papers - n_train
    bert_pred = pd.read_pickle('./data/valid_extral_withlogits_all.pickle')
    bert_pred['rank'] = bert_pred.groupby('idx')['pred'].rank(ascending=False)
    bert_pred['rank'] = bert_pred['rank'].apply(lambda x: int(x))
    papers_train = papers[:n_train]
    papers_valid = papers[n_train:]

    pids_train = {p["_id"] for p in papers_train}
    pids_valid = {p["_id"] for p in papers_valid}

    in_dir = join(data_dir, "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        if f.endswith(".xml"):
            files.append(f)

    pid_to_source_titles = dd(list)
    for paper in tqdm(papers):
        pid = paper["_id"]
        for ref in paper["refs_trace"]:
            pid_to_source_titles[pid].append(ref["title"].lower())
    model = SentenceTransformer('/home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/all-MiniLM-L6-v2')
    # files = sorted(files)
    # for file in tqdm(files):
    total = []
    cnt_train, cnt_valid = 0, 0
    ref_and_true_label = []
    for cur_pid in tqdm(pids_train):
        # cur_pid = file.split(".")[0]
        # if cur_pid not in pids_train and cur_pid not in pids_valid:
        # continue
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
        titles = []
        bids = []
        for bid in bid_to_title:
            titles.append(bid_to_title[bid])
            bids.append(bid)
        titles_embed = model.encode(titles, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        true_title = []
        for label_title in source_titles:
            true_title.append(label_title)
        true_titles_embed = model.encode(true_title, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        index = faiss.IndexFlatIP(384)
        index.add(titles_embed)
        prob, idx = index.search(true_titles_embed, 1)
        for i in range(idx.shape[0]):
            if prob[i][0] < 0.8:
                continue
            cur_pos_bib.add(bids[idx[i][0]])
            ref_and_true_label.append({'ref': titles[idx[i][0]], 'label': true_title[i]})

        cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib
        total.append(len(source_titles) - len(cur_pos_bib))
        if len(source_titles) - len(cur_pos_bib) != 0:
            a = 1
        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue

        n_pos = len(cur_pos_bib)
        n_neg = min(n_pos * 10, len(cur_neg_bib))
        cur_neg_bib_sample = np.random.choice(list(cur_neg_bib), n_neg, replace=True)

        title = bs.find('titleStmt').find('title').text
        text = "Title:\n"
        text += title + '\n'
        # 提取摘要
        abstract = bs.find('abstract')
        if abstract:
            text += "Abstract:\n"
            for p in abstract.find_all('p'):
                text += p.text + '\n'

        # 提取正文
        body = bs.find('body')
        text += 'Body:\n'
        bodys = []
        if body:
            for div in body.find_all('div'):
                texts = []
                for tag in div.descendants:
                    if tag.name:
                        texts.append(tag.text)
                bodys.append(''.join(texts) + '\n')

        references = bs.find('listBibl').find_all('biblStruct')
        ref_text = 'References:\n'
        cut = bert_pred[bert_pred['idx'] == cur_pid].reset_index(drop=True)

        # 按照文档中定义的顺序（即XML中的出现顺序）遍历参考文献
        for i, ref in enumerate(references):
            # 提取作者信息
            authors = [author.text for author in ref.find_all('persName')]
            author_str = ' '.join(authors)

            # 提取标题信息
            title = ref.find('title').text if ref.find('title') else 'No title'

            # 打印参考文献序号和信息
            rank = cut[cut['bib'] == f"b{i}"].reset_index(drop=True)
            rank = rank['rank'][0]
            ref_text += f"{i}, {title}, {author_str}.This ref paper's rank is {rank}.\n"

        for bib in cur_pos_bib:
            bid_title = bid_to_title[bib]
            rank = cut[cut['bib'] == bib].reset_index(drop=True)
            assert len(rank) == 1
            rank = rank['rank'][0]
            cur_context = "You are a computer thesis expert, and you have to judge whether a reference is the source paper of the original text based on the original text of the paper."
            cur_context += """The following points define whether a reference is a source paper:\nIs the main idea of paper p inspired by the reference？\nIs the core method of paper p derived from the reference？\nIs the reference essential for paper p? Without the work of this reference, paper p cannot be completed.\n"""
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += text
            for body in bodys:
                cur_context += body
                cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += ref_text
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            x_train.append(cur_context)
            y_train.append(1)
            idx_train.append(cur_pid)

        for bib in cur_neg_bib:
            rank = cut[cut['bib'] == bib].reset_index(drop=True)
            assert len(rank) == 1
            rank = rank['rank'][0]
            cur_context = "You are a computer thesis expert, and you have to judge whether a reference is the source paper of the original text based on the original text of the paper."
            cur_context += """The following points define whether a reference is a source paper:\nIs the main idea of paper p inspired by the reference？\nIs the core method of paper p derived from the reference？\nIs the reference essential for paper p? Without the work of this reference, paper p cannot be completed.\n"""
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += text
            for body in bodys:
                cur_context += body
                cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += ref_text
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            x_train.append(cur_context)
            y_train.append(0)
            idx_train.append(cur_pid)

    print(f"Train miss label:{np.sum(total)}, mean:{np.mean(total)}")
    print(f"Train total pos label:{np.sum(y_train)}")
    print(f"Train miss author's paper:{cnt_train}")
    total = []
    for cur_pid in tqdm(pids_valid):
        # cur_pid = file.split(".")[0]
        # if cur_pid not in pids_train and cur_pid not in pids_valid:
        # continue
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
        titles = []
        bids = []
        for bid in bid_to_title:
            titles.append(bid_to_title[bid])
            bids.append(bid)
        titles_embed = model.encode(titles, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        true_title = []
        for label_title in source_titles:
            true_title.append(label_title)
        true_titles_embed = model.encode(true_title, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        index = faiss.IndexFlatIP(384)
        index.add(titles_embed)
        prob, idx = index.search(true_titles_embed, 1)
        for i in range(idx.shape[0]):
            if prob[i][0] < 0.8:
                continue
            cur_pos_bib.add(bids[idx[i][0]])
            ref_and_true_label.append({'ref': titles[idx[i][0]], 'label': true_title[i]})
        total.append(len(source_titles) - len(cur_pos_bib))

        cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib

        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue

        n_pos = len(cur_pos_bib)
        n_neg = n_pos * 10

        title = bs.find('titleStmt').find('title').text
        text = "Title:\n"
        text += title + '\n'
        # 提取摘要
        abstract = bs.find('abstract')
        if abstract:
            text += "Abstract:\n"
            for p in abstract.find_all('p'):
                text += p.text + '\n'

        # 提取正文
        body = bs.find('body')
        text += 'Body:\n'
        bodys = []
        if body:
            for div in body.find_all('div'):
                texts = []
                for tag in div.descendants:
                    if tag.name:
                        texts.append(tag.text)
                bodys.append(''.join(texts) + '\n')

        references = bs.find('listBibl').find_all('biblStruct')
        ref_text = 'References:\n'
        # 按照文档中定义的顺序（即XML中的出现顺序）遍历参考文献
        for i, ref in enumerate(references):
            # 提取作者信息
            authors = [author.text for author in ref.find_all('persName')]
            author_str = ' '.join(authors)

            # 提取标题信息
            title = ref.find('title').text if ref.find('title') else 'No title'

            # 打印参考文献序号和信息
            # print(f'{i}. {author_str}, "{title}"')
            ref_text += f"{i}, {title}, {author_str}\n"
        cut = bert_pred[bert_pred['idx'] == cur_pid].reset_index(drop=True)
        for bib in cur_pos_bib:
            rank = cut[cut['bib'] == bib].reset_index(drop=True)
            assert len(rank) == 1
            rank = rank['rank'][0]
            cur_context = "You are a computer thesis expert, and you have to judge whether a reference is the source paper of the original text based on the original text of the paper."
            cur_context += """The following points define whether a reference is a source paper:\nIs the main idea of paper p inspired by the reference？\nIs the core method of paper p derived from the reference？\nIs the reference essential for paper p? Without the work of this reference, paper p cannot be completed.\n"""
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += text
            for body in bodys:
                cur_context += body
                cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += ref_text
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            x_valid.append(cur_context)
            y_valid.append(1)
            idx_valid.append(cur_pid)

        for bib in cur_neg_bib:
            rank = cut[cut['bib'] == bib].reset_index(drop=True)
            assert len(rank) == 1
            rank = rank['rank'][0]
            cur_context = "You are a computer thesis expert, and you have to judge whether a reference is the source paper of the original text based on the original text of the paper."
            cur_context += """The following points define whether a reference is a source paper:\nIs the main idea of paper p inspired by the reference？\nIs the core method of paper p derived from the reference？\nIs the reference essential for paper p? Without the work of this reference, paper p cannot be completed.\n"""
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += text
            for body in bodys:
                cur_context += body
                cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += ref_text
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            x_valid.append(cur_context)
            y_valid.append(0)
            idx_valid.append(cur_pid)
    print(f"Val miss label:{np.sum(total)}, mean:{np.mean(total)}")
    print(f"Val total pos label:{np.sum(y_valid)}")
    print(f"Val miss author's paper:{cnt_valid}")

    print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid))

    train = pd.DataFrame({'text': x_train, 'label': y_train, 'idx': idx_train})
    valid = pd.DataFrame({'text': x_valid, 'label': y_valid, 'idx': idx_valid})

    # train.to_pickle('./data/llm_train.pickle')
    # valid.to_pickle('./data/llm_valid.pickle')
    alls = pd.concat([train, valid], axis=0).reset_index(drop=True)
    alls.to_pickle('./data/llm_notsample_all_enhance_attention_purelabel.pickle')
    ref_and_true_label = pd.DataFrame(ref_and_true_label)
    ref_and_true_label.to_csv('./data/ref_and_true_label.csv')


def prepare_llm_addbib_test_input():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    idx_train = []
    idx_valid = []
    title_train = []
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")
    sub_example_dict = utils.load_json(data_dir, "submission_example_test.json")
    with open('./data/submission.json', 'r') as f:
        bert_pred = json.load(f)

    n_papers = len(papers)
    papers = sorted(papers, key=lambda x: x["_id"])
    xml_dir = join(data_dir, "paper-xml")
    sub_dict = {}
    total = 0
    source_title = []
    occur_cnts = []
    miss = 0
    bibs = []
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

        # total += n_refs
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]

        y_score = [0] * n_refs
        bib_to_contexts = utils.find_bib_context(xml)

        assert len(sub_example_dict[cur_pid]) == n_refs
        pp = bert_pred[cur_pid]
        temp = {value: i + 1 for i, value in enumerate(sorted(set(pp), reverse=True))}
        ranks = [temp[item] for item in pp]

        title = bs.find('titleStmt').find('title').text
        text = "Title:\n"
        text += title + '\n'
        # 提取摘要
        abstract = bs.find('abstract')
        if abstract:
            text += "Abstract:\n"
            for p in abstract.find_all('p'):
                text += p.text + '\n'

        # 提取正文
        body = bs.find('body')
        text += 'Body:\n'
        if body:
            for div in body.find_all('div'):
                texts = []
                for tag in div.descendants:
                    if tag.name:
                        if tag.attrs and 'type' in tag.attrs.keys() and tag.attrs['type'] == 'bibr':
                            try:
                                texts.append(tag.attrs['target'] + ' ')
                            except Exception:
                                a = 1
                        texts.append(tag.text)
                text += ''.join(texts) + '\n'

        references = bs.find('listBibl').find_all('biblStruct')
        text += 'References:\n'
        # 按照文档中定义的顺序（即XML中的出现顺序）遍历参考文献
        for i, ref in enumerate(references):
            # 提取作者信息
            authors = [author.text for author in ref.find_all('persName')]
            author_str = ' '.join(authors)

            # 提取标题信息
            title = ref.find('title').text if ref.find('title') else 'No title'

            text += f"{i}, {title}, {author_str}.\n"

        for i, bib in enumerate(bib_sorted):
            rank = ranks[i]
            occur_cnt = max(len(bib_to_contexts[bib]) - 1, 0)
            bibs.append(bib)
            cur_context = "You are a computer thesis expert, and you have to judge whether a reference is the source paper of the original text based on the original text of the paper."
            cur_context += """The following points define whether a reference is a source paper:\nIs the main idea of paper p inspired by the reference？\nIs the core method of paper p derived from the reference？\nIs the reference essential for paper p? Without the work of this reference, paper p cannot be completed.\n"""
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"
            cur_context += text
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"

            occur_cnts.append(occur_cnt)
            x_train.append(cur_context)
            title_train.append(bid_to_title[bib])
            idx_train.append(cur_pid)
            source_title.append(paper['title'])
    print(miss)
    test = pd.DataFrame(
        {'text': x_train, 'idx': idx_train, 'ref_title': title_train, 'source_title': source_title, 'bib': bibs,
         'occur_cnt': occur_cnts})
    test.to_pickle('./data/llm_final_title_addbib.pickle')


def prepare_llm_test_input():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    idx_train = []
    idx_valid = []
    title_train = []
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")
    sub_example_dict = utils.load_json(data_dir, "submission_example_test.json")
    with open('./data/submission.json', 'r') as f:
        bert_pred = json.load(f)

    n_papers = len(papers)
    papers = sorted(papers, key=lambda x: x["_id"])
    xml_dir = join(data_dir, "paper-xml")
    sub_dict = {}
    total = 0
    source_title = []
    occur_cnts = []
    ref_title_alltext = []
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
        bid_to_title_alltext = {}
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
            bid_to_title_alltext[bid] = ref.text
            if b_idx > n_refs:
                n_refs = b_idx

        # total += n_refs
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]

        y_score = [0] * n_refs
        bib_to_contexts = utils.find_bib_context(xml)

        assert len(sub_example_dict[cur_pid]) == n_refs
        pp = bert_pred[cur_pid]
        temp = {value: i + 1 for i, value in enumerate(sorted(set(pp), reverse=True))}
        ranks = [temp[item] for item in pp]

        title = bs.find('titleStmt').find('title').text
        text = "Title:\n"
        text += title + '\n'
        # 提取摘要
        abstract = bs.find('abstract')
        if abstract:
            text += "Abstract:\n"
            for p in abstract.find_all('p'):
                text += p.text + '\n'

        # 提取正文
        body = bs.find('body')
        text += 'Body:\n'
        if body:
            for div in body.find_all('div'):
                texts = []
                for tag in div.descendants:
                    if tag.name:
                        texts.append(tag.text)
                text += ''.join(texts) + '\n'

        references = bs.find('listBibl').find_all('biblStruct')
        text += 'References:\n'
        # 按照文档中定义的顺序（即XML中的出现顺序）遍历参考文献
        for i, ref in enumerate(references):
            # 提取作者信息
            authors = [author.text for author in ref.find_all('persName')]
            author_str = ' '.join(authors)

            # 提取标题信息
            title = ref.find('title').text if ref.find('title') else 'No title'

            text += f"{i}, {title}, {author_str}.\n"

        for i, bib in enumerate(bib_sorted):
            rank = ranks[i]
            occur_cnt = max(len(bib_to_contexts[bib]) - 1, 0)

            cur_context = "You are a computer thesis expert, and you have to judge whether a reference is the source paper of the original text based on the original text of the paper."
            cur_context += """The following points define whether a reference is a source paper:\nIs the main idea of paper p inspired by the reference？\nIs the core method of paper p derived from the reference？\nIs the reference essential for paper p? Without the work of this reference, paper p cannot be completed.\n"""
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"
            cur_context += text
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"

            occur_cnts.append(occur_cnt)
            x_train.append(cur_context)
            title_train.append(bid_to_title[bib])
            idx_train.append(cur_pid)
            source_title.append(paper['title'])
            ref_title_alltext.append(bid_to_title_alltext[bib])
    print(miss)
    test = pd.DataFrame({'text': x_train, 'idx': idx_train, 'ref_title': title_train, 'source_title': source_title,
                         'occur_cnt': occur_cnts, 'ref_title_alltext': ref_title_alltext})
    test.to_pickle('./data/llm_final_title.pickle')


def prepare_llm_alltext_input():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    idx_train = []
    idx_valid = []
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    n_papers = len(papers)
    papers = sorted(papers, key=lambda x: x["_id"])
    n_train = int(n_papers * 3 / 4)
    # n_valid = n_papers - n_train
    bert_pred = pd.read_pickle('./data/valid_extral_withlogits_all.pickle')
    bert_pred['rank'] = bert_pred.groupby('idx')['pred'].rank(ascending=False)
    bert_pred['rank'] = bert_pred['rank'].apply(lambda x: int(x))
    papers_train = papers[:n_train]
    papers_valid = papers[n_train:]

    pids_train = {p["_id"] for p in papers_train}
    pids_valid = {p["_id"] for p in papers_valid}

    in_dir = join(data_dir, "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        if f.endswith(".xml"):
            files.append(f)

    pid_to_source_titles = dd(list)
    for paper in tqdm(papers):
        pid = paper["_id"]
        for ref in paper["refs_trace"]:
            pid_to_source_titles[pid].append(ref["title"].lower())

    # files = sorted(files)
    # for file in tqdm(files):
    total = []
    cnt_train, cnt_valid = 0, 0
    for cur_pid in tqdm(pids_train):
        # cur_pid = file.split(".")[0]
        # if cur_pid not in pids_train and cur_pid not in pids_valid:
        # continue
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
        if not flag:
            continue

        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue

        n_pos = len(cur_pos_bib)
        n_neg = min(n_pos * 10, len(cur_neg_bib))
        cur_neg_bib_sample = np.random.choice(list(cur_neg_bib), n_neg, replace=True)

        text = xml
        cut = bert_pred[bert_pred['idx'] == cur_pid].reset_index(drop=True)

        for bib in cur_pos_bib:
            bid_title = bid_to_title[bib]
            rank = cut[cut['bib'] == bib].reset_index(drop=True)
            assert len(rank) == 1
            rank = rank['rank'][0]
            cur_context = "You are a computer thesis expert, and you have to judge whether a reference is the source paper of the original text based on the original text of the paper."
            cur_context += """The following points define whether a reference is a source paper:\nIs the main idea of paper p inspired by the reference？\nIs the core method of paper p derived from the reference？\nIs the reference essential for paper p? Without the work of this reference, paper p cannot be completed.\n"""
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += text.replace('\t', '')[:100000]
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            x_train.append(cur_context)
            y_train.append(1)
            idx_train.append(cur_pid)

        for bib in cur_neg_bib:
            rank = cut[cut['bib'] == bib].reset_index(drop=True)
            assert len(rank) == 1
            rank = rank['rank'][0]
            cur_context = "You are a computer thesis expert, and you have to judge whether a reference is the source paper of the original text based on the original text of the paper."
            cur_context += """The following points define whether a reference is a source paper:\nIs the main idea of paper p inspired by the reference？\nIs the core method of paper p derived from the reference？\nIs the reference essential for paper p? Without the work of this reference, paper p cannot be completed.\n"""
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += text.replace('\t', '')[:100000]
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            x_train.append(cur_context)
            y_train.append(0)
            idx_train.append(cur_pid)

    print(f"Train miss label:{np.sum(total)}, mean:{np.mean(total)}")
    print(f"Train total pos label:{np.sum(y_train)}")
    print(f"Train miss author's paper:{cnt_train}")
    total = []
    for cur_pid in tqdm(pids_valid):
        # cur_pid = file.split(".")[0]
        # if cur_pid not in pids_train and cur_pid not in pids_valid:
        # continue
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
        total.append(len(source_titles) - len(cur_pos_bib))

        cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib

        if not flag:
            continue

        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue

        n_pos = len(cur_pos_bib)
        n_neg = n_pos * 10

        text = xml

        cut = bert_pred[bert_pred['idx'] == cur_pid].reset_index(drop=True)
        for bib in cur_pos_bib:
            rank = cut[cut['bib'] == bib].reset_index(drop=True)
            assert len(rank) == 1
            rank = rank['rank'][0]
            cur_context = "You are a computer thesis expert, and you have to judge whether a reference is the source paper of the original text based on the original text of the paper."
            cur_context += """The following points define whether a reference is a source paper:\nIs the main idea of paper p inspired by the reference？\nIs the core method of paper p derived from the reference？\nIs the reference essential for paper p? Without the work of this reference, paper p cannot be completed.\n"""
            # cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += text.replace('\t', '')[:80000]
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            x_valid.append(cur_context)
            y_valid.append(1)
            idx_valid.append(cur_pid)

        for bib in cur_neg_bib:
            rank = cut[cut['bib'] == bib].reset_index(drop=True)
            assert len(rank) == 1
            rank = rank['rank'][0]
            cur_context = "You are a computer thesis expert, and you have to judge whether a reference is the source paper of the original text based on the original text of the paper."
            cur_context += """The following points define whether a reference is a source paper:\nIs the main idea of paper p inspired by the reference？\nIs the core method of paper p derived from the reference？\nIs the reference essential for paper p? Without the work of this reference, paper p cannot be completed.\n"""
            # cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += text.replace('\t', '')[:80000]
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            cur_context += f"The number of this reference paper is {bib}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(cut)}\n"
            x_valid.append(cur_context)
            y_valid.append(0)
            idx_valid.append(cur_pid)
    print(f"Val miss label:{np.sum(total)}, mean:{np.mean(total)}")
    print(f"Val total pos label:{np.sum(y_valid)}")
    print(f"Val miss author's paper:{cnt_valid}")

    print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid))

    train = pd.DataFrame({'text': x_train, 'label': y_train, 'idx': idx_train})
    valid = pd.DataFrame({'text': x_valid, 'label': y_valid, 'idx': idx_valid})

    # train.to_pickle('./data/llm_train_alltext_nobert.pickle')
    # valid.to_pickle('./data/llm_valid_alltext_nobert.pickle')
    alls = pd.concat([train, valid], axis=0).reset_index(drop=True)
    alls.to_pickle('./data/llm_notsample_all_alltext.pickle')


def prepare_llm_alltext_test_input():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    idx_train = []
    idx_valid = []
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_valid_wo_ans.json")
    sub_example_dict = utils.load_json(data_dir, "submission_example_valid.json")
    with open('./data/submission.json', 'r') as f:
        bert_pred = json.load(f)
    n_papers = len(papers)
    papers = sorted(papers, key=lambda x: x["_id"])
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

        # total += n_refs
        miss += (n_refs - len(bid_to_title.keys()))
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]

        y_score = [0] * n_refs

        assert len(sub_example_dict[cur_pid]) == n_refs
        pp = bert_pred[cur_pid]
        temp = {value: i + 1 for i, value in enumerate(sorted(set(pp), reverse=True))}
        ranks = [temp[item] for item in pp]
        text = xml
        for i, bib in enumerate(bib_sorted):
            rank = ranks[i]
            cur_context = "You are a computer thesis expert, and you have to judge whether a reference is the source paper of the original text based on the original text of the paper."
            cur_context += """The following points define whether a reference is a source paper:\nIs the main idea of paper p inspired by the reference？\nIs the core method of paper p derived from the reference？\nIs the reference essential for paper p? Without the work of this reference, paper p cannot be completed.\n"""
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"
            cur_context += text[:100000]
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"
            cur_context += f"The number of this reference paper is {bib[1:]}, and the name of this reference paper is <{bid_to_title[bib]}>.The rank of this paper is {rank}, total rank is {len(ranks)}\n"

            x_train.append(cur_context)
            idx_train.append(cur_pid)
    test = pd.DataFrame({'text': x_train, 'idx': idx_train})
    test.to_pickle('./data/llm_all_text_test.pickle')


def prepare_llm_pretrain_input():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    idx_train = []
    idx_valid = []
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    xml_dir = join(data_dir, "paper-xml")
    listdir = os.listdir(xml_dir)
    for paper in tqdm(listdir):
        if 'xml' not in paper:
            continue
        cur_pid = paper.split('.')[0]
        file = join(xml_dir, cur_pid + ".xml")
        f = open(file, encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()
        title = bs.find('titleStmt').find('title').text
        text = "Title:\n"
        text += title + '\n'
        # 提取摘要
        abstract = bs.find('abstract')
        if abstract:
            text += "Abstract:\n"
            for p in abstract.find_all('p'):
                text += p.text + '\n'

        # 提取正文
        body = bs.find('body')
        text += 'Body:\n'
        if body:
            for div in body.find_all('div'):
                texts = []
                for tag in div.descendants:
                    if tag.name:
                        texts.append(tag.text)
                text += ''.join(texts) + '\n'

        references = bs.find('listBibl').find_all('biblStruct')
        text += 'References:\n'
        # 按照文档中定义的顺序（即XML中的出现顺序）遍历参考文献
        for i, ref in enumerate(references):
            # 提取作者信息
            authors = [author.text for author in ref.find_all('persName')]
            author_str = ' '.join(authors)

            # 提取标题信息
            title = ref.find('title').text if ref.find('title') else 'No title'

            # 打印参考文献序号和信息
            # print(f'{i}. {author_str}, "{title}"')
            text += f"{i}, {title}, {author_str}\n"

        words = text.split(' ')
        chunks = []
        current_chunk = []

        for word in words:
            if len(current_chunk) + 1 <= 2048:
                current_chunk.append(word)
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]

        chunks.append(' '.join(current_chunk))  # 把最后一块加进去

        x_train.extend(chunks)
        idx_train.extend([cur_pid] * len(chunks))

    test = pd.DataFrame({'text': x_train, 'idx': idx_train})
    test.to_pickle('./data/llm_pretrain_text.pickle')


def prepare_3fold_bert_input():
    for fold in range(3):
        x_train = []
        y_train = []
        x_valid = []
        y_valid = []
        idx_train = []
        idx_valid = []
        title_valid = []
        occur_cnt = []
        bib_valid = []
        source_titleee = []
        data_dir = join(settings.DATA_TRACE_DIR, "PST")
        papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
        n_papers = len(papers)
        papers = sorted(papers, key=lambda x: x["_id"])
        n_train = int(n_papers * 2 / 3)
        # n_valid = n_papers - n_train

        blocks = np.array_split(papers, 3)
        papers_train, papers_valid = [], []
        for i in range(3):
            if i == fold:
                papers_valid.extend(blocks[i])
            else:
                papers_train.extend(blocks[i])

        pids_train = {p["_id"] for p in papers_train}
        pids_valid = {p["_id"] for p in papers_valid}

        in_dir = join(data_dir, "paper-xml")
        files = []
        for f in os.listdir(in_dir):
            if f.endswith(".xml"):
                files.append(f)

        pid_to_source_titles = dd(list)
        for paper in tqdm(papers):
            pid = paper["_id"]
            for ref in paper["refs_trace"]:
                pid_to_source_titles[pid].append(ref["title"].lower())

        # files = sorted(files)
        # for file in tqdm(files):
        total = []
        cnt_train, cnt_valid = 0, 0
        for cur_pid in tqdm(pids_train):
            # cur_pid = file.split(".")[0]
            # if cur_pid not in pids_train and cur_pid not in pids_valid:
            # continue
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
            cur_authors_paper = set()
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

            n_pos = len(cur_pos_bib)
            n_neg = n_pos * 10
            cur_neg_bib_sample = np.random.choice(list(cur_neg_bib), n_neg, replace=True)

            cur_x = x_train
            cur_y = y_train
            cur_idx = idx_train

            for bib in list(cur_pos_bib):
                cur_context = " ".join(bib_to_contexts[bib])
                cur_x.append(cur_context)
                cur_y.append(1)
                cur_idx.append(cur_pid)

            for bib in list(cur_neg_bib_sample):
                cur_context = " ".join(bib_to_contexts[bib])
                cur_x.append(cur_context)
                cur_y.append(0)
                cur_idx.append(cur_pid)

        print(f"Train miss label:{np.sum(total)}, mean:{np.mean(total)}")
        print(f"Train total pos label:{np.sum(y_train)}")
        print(f"Train miss author's paper:{cnt_train}")
        total = []
        for cur_pid in tqdm(pids_valid):
            # cur_pid = file.split(".")[0]
            # if cur_pid not in pids_train and cur_pid not in pids_valid:
            # continue
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
            total.append(len(source_titles) - len(cur_pos_bib))

            cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib

            if not flag:
                continue

            if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
                continue

            bib_to_contexts = utils.find_bib_context(xml)

            n_pos = len(cur_pos_bib)

            cur_x = x_valid
            cur_y = y_valid
            cur_idx = idx_valid
            title = bs.find('titleStmt').find('title').text
            for bib in list(cur_pos_bib):
                cur_context = " ".join(bib_to_contexts[bib])
                cur_x.append(cur_context)
                cur_y.append(1)
                cur_idx.append(cur_pid)
                title_valid.append(bid_to_title[bib])
                bib_valid.append(bib)
                source_titleee.append(title)
                occur_cnt.append(max(len(bib_to_contexts[bib]) - 1, 0))

            for bib in list(cur_neg_bib):
                cur_context = " ".join(bib_to_contexts[bib])
                cur_x.append(cur_context)
                cur_y.append(0)
                cur_idx.append(cur_pid)
                title_valid.append(bid_to_title[bib])
                bib_valid.append(bib)
                source_titleee.append(title)
                occur_cnt.append(max(len(bib_to_contexts[bib]) - 1, 0))
        print(f"Val miss label:{np.sum(total)}, mean:{np.mean(total)}")
        print(f"Val total pos label:{np.sum(y_valid)}")
        print(f"Val miss author's paper:{cnt_valid}")

        print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid))

        train = pd.DataFrame({'text':x_train, 'label':y_train, 'idx':idx_train})
        valid = pd.DataFrame({'text':x_valid, 'label':y_valid, 'idx':idx_valid, 'bib':bib_valid, 'ref_title':title_valid, 'occur_cnt':occur_cnt, 'source_title':source_titleee})
        train.to_pickle(f'./data/train_extral_fold{fold}.pickle')
        valid.to_pickle(f'./data/valid_extral_fold{fold}.pickle')


if __name__ == "__main__":
    prepare_3fold_bert_input()
