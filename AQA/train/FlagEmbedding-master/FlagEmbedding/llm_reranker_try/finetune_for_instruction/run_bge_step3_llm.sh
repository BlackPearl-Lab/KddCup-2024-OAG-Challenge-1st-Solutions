#!/bin/bash

PATH_PRE="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/zhouyang96"
source activate ${PATH_PRE}/conda_env/zy_from_chr_right_right


# model_name_or_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/Mistral-7B-v0.3
model_name_or_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/zhouyang96/zy_model_path/SOLAR-10.7B-v1.0
DATA_NAME="v3_round1_argu_qlora_recall_top_200_forrank_100_otherinfo_v2"
DATA_DIR=${PATH_PRE}/competiitons/kdd2024/task3/data/
MODEL_USE='SOLAR_5epoch_sm03_v3_round1_argu_qlora_recall_top_200_forrank_100_otherinfo_v2_argu_10'
OUTPUT=${PATH_PRE}/competiitons/kdd2024/task3/model_save/${MODEL_USE}
echo ${MODEL_USE}
# export CUDA_VISIBLE_DEVICES=5,6,7
torchrun --nproc_per_node 8 \
-m run \
--output_dir ${OUTPUT} \
--model_name_or_path ${model_name_or_path} \
--train_data ${DATA_DIR}${DATA_NAME}.jsonl \
--learning_rate 2e-4 \
--num_train_epochs 5 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--dataloader_drop_last True \
--query_max_len 512 \
--passage_max_len 512 \
--train_group_size 16 \
--logging_steps 1 \
--save_strategy epoch \
--save_steps 2 \
--save_total_limit 50 \
--ddp_find_unused_parameters False \
--gradient_checkpointing \
--deepspeed stage2.json \
--report_to "none" \
--warmup_ratio 0.05 \
--bf16 \
--use_lora True \
--lora_rank 32 \
--lora_alpha 64 \
--use_flash_attn False \
--target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj lm_head