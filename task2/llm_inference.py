# -*- coding: utf-8 -*-

import os
from peft import PeftModel,get_peft_model
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
import torch
from utils import IND4EVAL
import json
from accelerate import Accelerator
from tqdm import tqdm
import argparse

_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--lora_path', help='The path to the lora file',default="saved_dir/checkpoint-100")
parser.add_argument('--model_path',default='ZhipuAI/chatglm3-6b-32k')
parser.add_argument('--pub_path', help='The path to the pub file',default='test_pub.json')
parser.add_argument('--eval_path',default='eval_data.json')
parser.add_argument('--save_name', default='tmp')
parser.add_argument('--data_path', default='tmp')
parser.add_argument('--saved_dir',default='./eval_result')
args = parser.parse_args()

checkpoint = args.lora_path.split('/')[-1]

accelerator = Accelerator()
device = torch.device(0)

batch_size = 1

# model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_8bit=False, trust_remote_code=True).half()

q_config = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=_compute_dtype_map['bf16'])
config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                  config=config,
                                  quantization_config=q_config,
                                  trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = PeftModel.from_pretrained(model, args.lora_path)
print('done loading peft model')
YES_TOKEN_IDS = tokenizer.convert_tokens_to_ids("yes")
NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids("no")

import pandas as pd

valid = pd.read_pickle(args.data_path)
print(f"Valid size:{len(valid)}")
dev_texts = valid['process_text'].tolist()
dev_idx = valid['idx'].tolist()

eval_dataset = IND4EVAL(
    dev_texts,[i for i in range(len(dev_texts))],
    max_source_length = 31000,
    max_target_length = 16,
)
print('done reading dataset')

def collate_fn(batch):
    batch = {k: [item[k] for item in batch] for k in ('input_ids','idx')}
    batch_input = tokenizer(
        batch['input_ids'],
        padding='longest',
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=32768
    )
    return batch_input, batch['idx']

dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size = batch_size ,collate_fn=collate_fn)
val_data = accelerator.prepare_data_loader(dataloader, device_placement=True)
model = accelerator.prepare_model(model)
result = []
print('len val data: ', len(val_data))


YES_TOKEN_IDS = tokenizer.convert_tokens_to_ids("yes")
NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids("no")

with torch.no_grad():
    for index,batch in enumerate(tqdm(val_data)):
        batch_input, idx = batch
        response = model.generate(**batch_input, max_length=batch_input['input_ids'].shape[-1] + 1, return_dict_in_generate=True, output_scores=True)
        yes_prob, no_prob = response.scores[0][:,YES_TOKEN_IDS],response.scores[0][:,NO_TOKEN_IDS]
        logit = yes_prob/(yes_prob+no_prob)
        node_result = [[idx[i],logit[i].item()] for i in range(batch_size)]
        batch_result = accelerator.gather_for_metrics(node_result)
        if accelerator.is_main_process:
            result.extend(batch_result)

if accelerator.is_main_process:
    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)
    res_list = {}
    for i in result:
        [idx,logit] = i
        if idx not in res_list.keys():
            res_list[idx] = {}
        res_list[idx] = logit
    with open(f'{args.saved_dir}/{args.save_name}.json', 'w') as f:
        json.dump(res_list, f)
