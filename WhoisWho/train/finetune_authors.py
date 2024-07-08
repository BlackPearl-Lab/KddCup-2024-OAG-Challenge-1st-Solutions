#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""

# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
# Adapted from


import logging
import os
import sys
import json
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    AutoModelForCausalLM
)
from trainer import LoRATrainer
from arguments import ModelArguments, DataTrainingArguments, GLMTrainingArguments
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
import sys
sys.path.append('../utils/')
from utils_authors import INDDataSet, DataCollatorForIND

from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GLMTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.ddp_find_unused_parameters = False
    training_args.save_safetensors = False

    logging.basicConfig(

        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.eos_token # if the tokenizer not have pad_token
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True).cuda()
    # model = PeftModel.from_pretrained(model, model_args.lora_path)
    # model = model.merge_and_unload()
    
    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)

    with open(data_args.pub_data, "r", encoding="utf-8") as f:
        pub_data = json.load(f)
    with open(data_args.train_data, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    train_dataset = INDDataSet(
        (train_data, pub_data),
        tokenizer,
        data_args.max_source_length,
        data_args.max_target_length,
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_rank,
        target_modules=['query_key_value'],
        # target_modules=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['llama'],
        # target_modules=["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # target_modules=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['mistral'],
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
    )
    model = get_peft_model(model, peft_config).to("cuda")

    data_collator = DataCollatorForIND(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False,
    )

    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # breakpoint() #加断点进入调试
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.save_state()


if __name__ == "__main__":
    main()