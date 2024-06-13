#!/bin/bash
#!/bin/bash



PATH_PRE="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/zhouyang96"
source activate ${PATH_PRE}/conda_env/zy_from_chr_right_right

# DATA_NAME="bge_train_d1_mistrial_all"
DATA_NAME="v3_merge_123_new123"
DOC_NAME="AQA/pid_to_title_abs_new"
DATA_DIR=${PATH_PRE}/competiitons/kdd2024/task3/data/
MODEL_USE="v3_merge_123_new123"
ZERO_STAGE=2
OUTPUT=${PATH_PRE}/competiitons/kdd2024/task3/model_save/${MODEL_USE}_qlora


#模型地址
MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/SFR-Embedding-Mistral
# MODEL_PATH=${PATH_PRE}/competiitons/kdd2024/task3/model_save/second_SFR_fintune_first_SFR_fintune_bge_train_d1_mistrial_all_qlora_paper_19epoch_recall_top_100_qlora/epoch_19_model_merge
# s
mkdir -p ${OUTPUT}
echo ${ZERO_STAGE}
echo ${OUTPUT}

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
echo ${MASTER_PORT}
deepspeed  --master_port ${MASTER_PORT}  --include localhost:0,1,2,3,4,5,6,7 simcse_deepspeed_mistrial_qlora_argu.py \
       --project_name ${name}_${MODEL_USE} \
       --train_data ${DATA_DIR}${DATA_NAME}.jsonl \
       --doc_data ${DATA_DIR}${DOC_NAME}.json \
       --model_name_or_path ${MODEL_PATH} \
       --per_device_train_batch_size 4 \
       --per_device_eval_batch_size 4 \
       --train_group_size 4 \
       --gradient_accumulation_steps 1 \
       --query_max_len 256 \
       --passage_max_len 256 \
       --earystop 0 \
       --save_batch_steps 100000000000 \
       --eary_stop_epoch 5 \
       --save_per_epoch 1 \
       --num_train_epochs 20  \
       --learning_rate 1e-4 \
       --num_warmup_steps 100 \
       --weight_decay 0.01 \
       --lr_scheduler_type cosine \
       --seed 1234 \
       --zero_stage $ZERO_STAGE \
       --deepspeed \
       --output_dir $OUTPUT \
       --gradient_checkpointing