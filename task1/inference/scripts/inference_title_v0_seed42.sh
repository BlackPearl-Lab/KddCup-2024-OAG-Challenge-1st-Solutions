accelerate launch --num_processes 8 inference_titles.py \
    --lora_path ../output/chatglm_title \
    --model_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/chatglm3-6b-32k \
    --pub_path ../data/pid_to_info_all.json  \
    --eval_path ../data/IND-test-public/ind_test_author_filter_public.json \
    --saved_dir ../test_result \
    --seed 42 \
    --max_source_length 30000 \
    --max_target_length 16 \
    --test_score_file None \
    --save_name title_v0_seed42.json
    
    
    
    