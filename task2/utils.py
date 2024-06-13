from os.path import join
import json
import numpy as np
import pickle
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from datetime import datetime

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


def load_json(rfdir, rfname):
    logger.info('loading %s ...', rfname)
    with open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        data = json.load(rf)
        logger.info('%s loaded', rfname)
        return data


def dump_json(obj, wfdir, wfname):
    logger.info('dumping %s ...', wfname)
    with open(join(wfdir, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, indent=4, ensure_ascii=False)
    logger.info('%s dumped.', wfname)


def serialize_embedding(embedding):
    return pickle.dumps(embedding)


def deserialize_embedding(s):
    return pickle.loads(s)


def find_bib_context(xml, dist=200):
    bs = BeautifulSoup(xml, "xml")
    bib_to_context = dd(set)
    bibr_strs_to_bid_id = {}
    for item in bs.find_all(type='bibr'):
        if "target" not in item.attrs:
            continue
        bib_id = item.attrs["target"][1:]
        item_str = "<ref type=\"bibr\" target=\"{}\">{}</ref>".format(item.attrs["target"], item.get_text())
        bibr_strs_to_bid_id[item_str] = bib_id

    for item_str in bibr_strs_to_bid_id:
        bib_id = bibr_strs_to_bid_id[item_str]
        cur_bib_context_pos_start = [ii for ii in range(len(xml)) if xml.startswith(bib_id, ii)]
        for pos in cur_bib_context_pos_start:
            bib_to_context[bib_id].add(xml[pos - dist: pos + dist].replace("\n", " ").replace("\r", " ").strip())
    new_bib_to_context = dd(list)
    for item_str in bibr_strs_to_bid_id:
        bib_id = bibr_strs_to_bid_id[item_str]
        tmp = [f"The number of this reference paper is {bib_id}."] + list(bib_to_context[bib_id])
        new_bib_to_context[bib_id] = tmp
    return new_bib_to_context

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Log:
    def __init__(self, file_path):
        self.file_path = file_path
        self.f = open(file_path, 'w+')

    def log(self, s):
        self.f.write(str(datetime.now()) + "\t" + s + '\n')
        self.f.flush()


import json
from sklearn import metrics
import numpy as np
from torch.utils.data import Dataset
import random
import numpy as np
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy


class INDDeekspeekallinfoDataSet(Dataset):
    def __init__(self, data, tokenizer, max_source_length, max_target_length):
        super(INDDeekspeekallinfoDataSet, self).__init__()
        self.data = data.sample(len(data), random_state=0).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.instruct = "<|im_start|>user\n{}, \n###\nGive me an answer between 'yes' or 'no'.<|im_end|>\n<|im_start|>assistant\n"

        self.yes_token = self.tokenizer.encode(text='yes', add_special_tokens=False, truncation=True, )
        self.no_token = self.tokenizer.encode(text='no', add_special_tokens=False, truncation=True, )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        now_data = self.data.loc[index]
        text = now_data['process_text']
        context = self.instruct.format(text)
        input_ids = self.tokenizer.encode(text=context, add_special_tokens=True, truncation=True,
                                          max_length=32768)
        label_ids = self.yes_token if int(now_data['label']) else self.no_token
        input_ids = input_ids + label_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * (len(input_ids) - 2) + label_ids + [self.tokenizer.eos_token_id]

        return {
            "input_ids": input_ids,
            "labels": labels
        }


class INDallinfoDataSet(Dataset):
    def __init__(self, data, tokenizer, max_source_length, max_target_length):
        super(INDallinfoDataSet, self).__init__()
        self.data = data.sample(len(data), random_state=0).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        # self.instruct = "{}, \n###\nGive me an answer between 'yes' or 'no'."
        self.instruct = "<|im_start|>user\n{}, \n###\nGive me an answer between 'yes' or 'no'.<|im_end|>\n<|im_start|>assistant\n"

        self.yes_token = self.tokenizer.encode(text='yes', add_special_tokens=False, truncation=True, )
        self.no_token = self.tokenizer.encode(text='no', add_special_tokens=False, truncation=True, )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        now_data = self.data.loc[index]
        text = now_data['process_text']
        context = self.instruct.format(text)
        input_ids = self.tokenizer.encode(text=context, add_special_tokens=True, truncation=True,
                                          max_length=32768)
        label_ids = self.yes_token if int(now_data['label']) else self.no_token
        input_ids = input_ids + label_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * (len(input_ids) - 2) + label_ids + [self.tokenizer.eos_token_id]

        return {
            "input_ids": input_ids,
            "labels": labels
        }


class INDDataSet(Dataset):
    '''
        iteratively return the profile of each author
    '''

    def __init__(self, dataset, tokenizer, max_source_length, max_target_length):
        super(INDDataSet, self).__init__()
        self.texts, self.labels = dataset
        self.data = [{'text': text, 'label': label} for text, label in zip(self.texts, self.labels)]
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        random.shuffle(self.data)
        self.instruct = "{}, \n###\nGive me an answer between 'yes' or 'no'."

        self.yes_token = self.tokenizer.encode(text='yes', add_special_tokens=False, truncation=True, )
        self.no_token = self.tokenizer.encode(text='no', add_special_tokens=False, truncation=True, )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        now_data = self.data[index]
        text = now_data['text']
        label = now_data['label']
        context = self.instruct.format(text)

        input_ids = self.tokenizer.encode(text=context, add_special_tokens=True, truncation=True,
                                          max_length=self.max_source_length)
        label_ids = self.yes_token if int(label) else self.no_token
        input_ids = input_ids + label_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * (len(input_ids) - 2) + label_ids + [self.tokenizer.eos_token_id]

        return {
            "input_ids": input_ids,
            "labels": labels
        }


@dataclass
class DataCollatorForIND:
    """
        borrow and modified from transformers.DataCollatorForSeq2Seq
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        # breakpoint()
        features = self.tokenizer.pad(
            features,
            padding=True,
            max_length=max_label_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        # breakpoint() # [(len(features[i]['input_ids']),len(features[i]['labels'])) for i in range(4)]
        return features


class IND4EVAL(Dataset):
    def __init__(self, val_texts, val_idx, max_source_length, max_target_length):
        super(IND4EVAL, self).__init__()
        self.val_texts = val_texts
        self.val_idx = val_idx
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.instruct = "{}, \n###\nGive me an answer between 'yes' or 'no'."
        # self.instruct = "<s>[INST]{}, , \n###\nGive me an answer between 'yes' or 'no'.[/INST]"

    def __len__(self):
        return len(self.val_texts)

    def __getitem__(self, index):
        idx = self.val_idx[index]
        text = self.val_texts[index]
        context = self.instruct.format(text)
        return {
            "input_ids": context,
            'idx': idx
        }


class IND4EVALCLS(Dataset):
    def __init__(self, tokenizer, val_texts, val_idx, max_source_length, max_target_length):
        super(IND4EVALCLS, self).__init__()
        self.val_texts = val_texts
        self.tokenizer = tokenizer
        self.val_idx = val_idx
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        # self.instruct = "{}, \n###\nGive me an answer between 'yes' or 'no'."

    def __len__(self):
        return len(self.val_texts)

    def __getitem__(self, index):
        idx = self.val_idx[index]
        text = self.val_texts[index]
        input_ids = self.tokenizer.encode(text=text, add_special_tokens=True, truncation=True,
                                          max_length=self.max_source_length)
        input_ids = input_ids + self.tokenizer.encode('Give me an answer', add_special_tokens=False)
        return {
            "input_ids": input_ids,
            'idx': idx
        }

    @staticmethod
    @staticmethod
    def collate(batch):
        batch_input, batch_labels = [], []
        for item in batch:
            batch_input.append(item['input_ids'])
            batch_labels.append(item['idx'])
        batch_input = torch.tensor(batch_input, dtype=torch.long)

        return batch_input, batch_labels


class INDCLSDataSet(Dataset):
    '''
        iteratively return the profile of each author
    '''

    def __init__(self, dataset, tokenizer, max_source_length, max_target_length):
        super(INDCLSDataSet, self).__init__()
        self.texts, self.labels = dataset
        self.data = [{'text': text, 'label': label} for text, label in zip(self.texts, self.labels)]
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        now_data = self.data[index]
        text = now_data['text']
        label = now_data['label']
        input_ids = self.tokenizer.encode(text=text, add_special_tokens=True, truncation=True,
                                          max_length=self.max_source_length)
        input_ids = input_ids + self.tokenizer.encode('Give me an answer', add_special_tokens=False)

        return {
            "input_ids": input_ids,
            "labels": label
        }

    @staticmethod
    def collate(batch):
        batch_input, batch_labels = [], []
        for item in batch:
            batch_input.append(item['input_ids'])
            batch_labels.append(item['labels'])
        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float)

        return {'input_ids': batch_input, 'labels': batch_labels}