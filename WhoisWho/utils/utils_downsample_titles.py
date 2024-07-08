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
from imblearn.under_sampling import RandomUnderSampler

class INDDataSet(Dataset):
    '''
        iteratively return the profile of each author 
    '''
    def __init__(self, dataset, tokenizer, max_source_length, max_target_length):
        super(INDDataSet, self).__init__()
        self.author, self.pub = dataset  
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        author_keys = self.author.keys()
        train_keys = []
        labels = []
        for key in author_keys :
            for i in self.author[key]['outliers']:
                train_keys.append({
                    "pub": i,
                    "author": key,
                    "label": 0
                }) 
                labels.append(0)
            for i in self.author[key]['normal_data']:
                train_keys.append({
                    "pub": i,
                    "author": key,
                    "label": 1
                })
                labels.append(1)
        rus = RandomUnderSampler(random_state=0)
        keys_ids = list(range(0,len(train_keys)))
        keys_ids = [ [x, 0] for x in keys_ids ]
        sampled_keys,_ = rus.fit_resample(keys_ids, labels)
        self.train_keys = [train_keys[i[0]] for i in sampled_keys]
        # self.train_keys = train_keys
        random.shuffle(self.train_keys)
        self.instruct = "Identify the abnormal text from the text collection according to the following rules:\n Here is a collection of paper titles: \n ### {} \n ### Does the paper title ### {} ### belong to the main part of these papers, give me an answer between 'yes' or 'no'."

        self.yes_token = self.tokenizer.encode(text='yes', add_special_tokens=False, truncation=True,)
        self.no_token = self.tokenizer.encode(text='no', add_special_tokens=False, truncation=True,)


    def __len__(self):
        return len(self.train_keys)

    def __getitem__(self, index):
        profile = self.author[self.train_keys[index]['author']]['normal_data'] +self.author[self.train_keys[index]['author']]['outliers']
        profile = [self.pub[p]['title'] for p in profile if p != self.train_keys[index]['pub']] #delete disambiguate paper
        random.shuffle(profile)
        # breakpoint()
        # limit context token lenth up to max_len - 500
        tokenized_profile = [self.tokenizer.tokenize(i) for i in profile]
        len_profile = [len(i) for i in tokenized_profile]
        sum_len = sum(len_profile)
        if sum_len> self.max_source_length-500: # left 500 for the instruction templete
            total_len = 0
            p = 0   
            while total_len < self.max_source_length-500 and p < sum_len:
                total_len += len_profile[p]
                p += 1
            profile = profile[:p-1]

        profile_text = ' # '.join(profile)
        title = self.pub[self.train_keys[index]['pub']]['title']
        title = title if len(self.tokenizer.tokenize(title))<200 else ' '.join(title.split(' ')[:100]) #limit the disambiguate paper title token lenth
        context = self.instruct.format(profile_text,title)
        
        input_ids = self.tokenizer.encode(text=context, add_special_tokens=True, truncation=True, max_length=self.max_source_length)
        label_ids = self.yes_token if self.train_keys[index]['label'] else self.no_token
        input_ids = input_ids + label_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * (len(input_ids)-2) + label_ids + [self.tokenizer.eos_token_id]

        return {
            "input_ids":input_ids,
            "labels":labels,
            "author":self.train_keys[index]['author'],
            "pub":self.train_keys[index]['pub'],
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
    def __init__(self, dataset, tokenizer, max_source_length, max_target_length):
        super(IND4EVAL, self).__init__()
        self.author, self.pub = dataset  
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        author_keys = self.author.keys()

        self.val_set = []
        if 'normal_data' in self.author[list(author_keys)[0]]:
            for key in author_keys:   
                for pub_key in self.author[key]['normal_data']:   
                    self.val_set.append({
                        'pub':pub_key,
                        'author':key,
                        'label':1
                    }) 
                for pub_key in self.author[key]['outliers']:
                    self.val_set.append({
                        'pub':pub_key,
                        'author':key,
                        'label':0
                    }) 
        elif 'papers' in self.author[list(author_keys)[0]]:
            for key in author_keys:   
                for pub_key in self.author[key]['papers']:   
                    self.val_set.append({
                        'pub':pub_key,
                        'author':key,
                    }) 
        self.instruct = "Identify the abnormal text from the text collection according to the following rules:\n Here is a collection of paper titles: \n ### {} \n ### Does the paper title ### {} ### belong to the main part of these papers, give me an answer between 'yes' or 'no'."

    def __len__(self):
        return len(self.val_set)
    
    def __getitem__(self, index):
        if "normal_data" in self.author[self.val_set[index]['author']]:
            profile = self.author[self.val_set[index]['author']]['normal_data'] +self.author[self.val_set[index]['author']]['outliers']
        elif "papers" in self.author[self.val_set[index]['author']]:
            profile = self.author[self.val_set[index]['author']]['papers']
        else:
            raise("No profile found")
        profile = [self.pub[p]['title'] for p in profile if p != self.val_set[index]['pub']] #delete disambiguate paper
        random.shuffle(profile)

        tokenized_profile = [self.tokenizer.tokenize(i) for i in profile]
        len_profile = [len(i) for i in tokenized_profile]
        sum_len = sum(len_profile)
        if sum_len> self.max_source_length-500:
            total_len = 0
            p = 0   
            while total_len < self.max_source_length-500 and p < sum_len:
                total_len += len_profile[p]
                p += 1
            profile = profile[:p-1]

        profile_text = ' # '.join(profile)
        title = self.pub[self.val_set[index]['pub']]['title']
        title = title if len(self.tokenizer.tokenize(title))<200 else ' '.join(title.split(' ')[:100]) 
        context = self.instruct.format(profile_text,title)
        return {
            "input_ids":context,
            "author":self.val_set[index]['author'],
            "pub":self.val_set[index]['pub'],
        }
