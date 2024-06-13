accelerate launch --main_process_port 8420 --num_processes 8 inference.py \
    --lora_path ./output/chatglm_lr1e-4_rank128_chat_more_info_epoch2_addauthor_and_cite_v31_retrain/checkpoint-1950 \
    --save_name addbib_addcite_step1950 \
    --model_path ZhipuAI/chatglm3-6b-32k