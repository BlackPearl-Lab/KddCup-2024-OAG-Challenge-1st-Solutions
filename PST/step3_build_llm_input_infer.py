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
    train.to_pickle('./data/llm_notsample_with_allinfo_addbib_addcite.pickle')




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





if __name__ == "__main__":
    prepare_llm_test_input()
    prepare_llm_addbib_test_input()

