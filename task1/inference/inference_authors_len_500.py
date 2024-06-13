# -*- coding: utf-8 -*-

import os
from peft import PeftModel,get_peft_model
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM,set_seed
import torch
import sys
sys.path.append('../utils/')
from utils_authors_len_500 import IND4EVAL
import json
from accelerate import Accelerator
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lora_path', help='The path to the lora file',default="saved_dir/checkpoint-100")
parser.add_argument('--model_path',default='ZhipuAI/chatglm3-6b-32k')
parser.add_argument('--pub_path', help='The path to the pub file',default='test_pub.json')
parser.add_argument('--eval_path',default='eval_data.json')
parser.add_argument('--saved_dir',default='../eval_result')
parser.add_argument('--test_score_file',default='../eval_result/merge_all_334.json')
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--max_source_length',type=int,default=30000)
parser.add_argument('--max_target_length',type=int,default=16)
parser.add_argument('--save_name',default='test_result.json')
args = parser.parse_args()

set_seed(args.seed)

checkpoint = args.lora_path.split('/')[-1]

accelerator = Accelerator()
device = torch.device(0)

batch_size = 1

model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_8bit=False, trust_remote_code=True).half()
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
lora_model = PeftModel.from_pretrained(model, args.lora_path).half()
print('done loading peft model')
YES_TOKEN_IDS = tokenizer.convert_tokens_to_ids("yes")
NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids("no")

with open(args.pub_path, "r" , encoding = "utf-8") as f:
    pub_data = json.load(f)
with open(args.eval_path, "r", encoding="utf-8") as f: 
    eval_data = json.load(f)
eval_dataset = IND4EVAL(
    (eval_data,pub_data),
    tokenizer,
    max_source_length = args.max_source_length,
    max_target_length = args.max_target_length,
    test_score_file = None if args.test_score_file == 'None' else args.test_score_file
) 
print('done reading dataset')

def collate_fn(batch):
    batch = {k: [item[k] for item in batch] for k in ('input_ids', 'author', 'pub')}
    batch_input = tokenizer(
        batch['input_ids'],
        padding='longest',
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return batch_input,batch['author'],batch['pub']

dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size = batch_size ,collate_fn=collate_fn)
val_data = accelerator.prepare_data_loader(dataloader, device_placement=True)
model = accelerator.prepare_model(model)
result = []
print('len val data: ', len(val_data))


YES_TOKEN_IDS = tokenizer.convert_tokens_to_ids("yes")
NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids("no")

with torch.no_grad():
    for index,batch in enumerate(tqdm(val_data)):
        batch_input, author, pub = batch
        response = model.generate(**batch_input, max_length=batch_input['input_ids'].shape[-1] + 16, return_dict_in_generate=True, output_scores=True)

        yes_prob, no_prob = response.scores[0][:,YES_TOKEN_IDS],response.scores[0][:,NO_TOKEN_IDS]
        logit = yes_prob/(yes_prob+no_prob)
        node_result = [(author[i],pub[i],logit[i].item()) for i in range(batch_size)]
        batch_result = accelerator.gather_for_metrics(node_result)
        if accelerator.is_main_process:
            result.extend(batch_result)

if accelerator.is_main_process: 
    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)
    res_list = {}
    for i in result:
        [aid,pid,logit] = i
        if aid not in res_list.keys():
            res_list[aid] = {}
        res_list[aid][pid] = logit
    save_path = os.path.join(args.saved_dir,args.save_name)
    with open(save_path, 'w') as f:
        json.dump(res_list, f)
