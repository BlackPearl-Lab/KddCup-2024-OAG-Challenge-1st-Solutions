accelerate launch --num_processes 8 inference_authors.py \
    --lora_path ../output/chatglm_author \
    --model_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/chatglm3-6b-32k \
    --pub_path ../data/pid_to_info_all.json  \
    --eval_path ../data/IND-test-public/ind_test_author_filter_public.json \
    --saved_dir ../test_result \
    --seed 42 \
    --max_source_length 32000 \
    --max_target_length 16 \
    --test_score_file ../test_result/merge_all_334.json \
    --save_name author_v2.json
    
    
    
    