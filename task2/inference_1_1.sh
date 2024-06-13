accelerate launch --main_process_port 8420 --num_processes 8 inference.py \
    --lora_path ./output/addbib_addcite/checkpoint-950 \
    --save_name addbib_addcite_step950 \
    --data_path ./data/llm_final_title_addbib_moreinfo_processtext.pickle \
    --model_path ZhipuAI/chatglm3-6b-32k