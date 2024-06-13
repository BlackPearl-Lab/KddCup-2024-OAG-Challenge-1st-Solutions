#!/bin/bash


################################## step1 get doc embeddings and test top recall ########################################
path_pre="./"
num=8 # Number of GPUs for prediction
model_path="../model_save/SFR-Embedding-Mistral"
rank_model_path="/home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/zhouyang96/zy_model_path/SOLAR-10.7B-v1.0"

model_version="v3_round1_argu_qlora"
save_name=${model_version}_doc
lora_path="../model_save/${model_version}/epoch_19_model/adapter.bin"
rank_lora_path="../model_save/SOLAR_5epoch_sm04_v3_round1_argu_qlora_recall_top_100_rank/"
python -u get_doc_embeddings.py ${save_name} ${path_pre} ${model_path} ${lora_path} ${num}
python -u get_test_recall.py ${model_version} ${model_path}
python -u get_test_rank_result.py ${model_version} ${rank_model_path} ${num} ${rank_lora_path}

model_version="v3_round2_argu_qlora_qlora"
save_name=${model_version}_doc
lora_path="../model_save/${model_version}/epoch_19_model/adapter.bin"
rank_lora_path="../model_save/SOLAR_5epoch_sm03_v3_round2_argu_qlora_qlora_recall_top_100_rank/checkpoint-635/"
python -u get_doc_embeddings.py ${save_name} ${path_pre} ${model_path} ${lora_path} ${num}
python -u get_test_recall.py ${model_version} ${model_path}
python -u get_test_rank_result.py ${model_version} ${rank_model_path} ${num} ${rank_lora_path}


model_version="v3_round3_argu_qlora_qlora"
save_name=${model_version}_doc
lora_path="../model_save/${model_version}/epoch_19_model/adapter.bin"
rank_lora_path="../model_save/SOLAR_5epoch_sm03_v3_round3_argu_qlora_qlora_recall_top_100_rank/"
python -u get_doc_embeddings.py ${save_name} ${path_pre} ${model_path} ${lora_path} ${num}
python -u get_test_recall.py ${model_version} ${model_path}
python -u get_test_rank_result.py ${model_version} ${rank_model_path} ${num} ${rank_lora_path}


model_version="v3_merge_123_top200_qlora"
save_name=${model_version}_doc
lora_path="../model_save/${model_version}/epoch_19_model/adapter.bin"
rank_lora_path="../model_save/SOLAR_5epoch_sm02_v3_merge_123_top100_rank_lora/"
python -u get_doc_embeddings.py ${save_name} ${path_pre} ${model_path} ${lora_path} ${num}
python -u get_test_recall.py ${model_version} ${model_path}
python -u get_test_rank_result.py ${model_version} ${rank_model_path} ${num} ${rank_lora_path}

model_version="v3_merge_123_new_1_qlora"
save_name=${model_version}_doc
lora_path="../model_save/${model_version}/epoch_19_model/adapter.bin"
rank_lora_path="../model_save/SOLAR_5epoch_sm03_v3_merge_123_new_1_qlora_top100_rank/"
python -u get_doc_embeddings.py ${save_name} ${path_pre} ${model_path} ${lora_path} ${num}
python -u get_test_recall.py ${model_version} ${model_path}
python -u get_test_rank_result.py ${model_version} ${rank_model_path} ${num} ${rank_lora_path}

model_version="v3_merge_123_new_2_qlora"
save_name=${model_version}_doc
lora_path="../model_save/${model_version}/epoch_19_model/adapter.bin"
rank_lora_path="../model_save/SOLAR_5epoch_sm03_v3_merge_123_new_2_qlora_top100_rank/"
python -u get_doc_embeddings.py ${save_name} ${path_pre} ${model_path} ${lora_path} ${num}
python -u get_test_recall.py ${model_version} ${model_path}
python -u get_test_rank_result.py ${model_version} ${rank_model_path} ${num} ${rank_lora_path}

model_version="v3_merge_123_new_3_qlora"
save_name=${model_version}_doc
lora_path="../model_save/${model_version}/epoch_19_model/adapter.bin"
rank_lora_path="../model_save/SOLAR_5epoch_sm03_v3_merge_123_new_3_qlora_top100_rank/"
python -u get_doc_embeddings.py ${save_name} ${path_pre} ${model_path} ${lora_path} ${num}
python -u get_test_recall.py ${model_version} ${model_path}
python -u get_test_rank_result.py ${model_version} ${rank_model_path} ${num} ${rank_lora_path}


################################# step2 rank avg ##########################
python -u rank_avg.py