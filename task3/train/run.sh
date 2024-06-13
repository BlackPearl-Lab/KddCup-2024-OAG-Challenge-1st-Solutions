
#!/bin/bash



path_pre="./"
num=8 # Number of GPUs for prediction
model_path="../model_save/SFR-Embedding-Mistral"
rank_model_path="/home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/zhouyang96/zy_model_path/SOLAR-10.7B-v1.0"



model_version="zero_round"
save_name=${model_version}_doc
lora_path="none"
rank_lora_path="../model_save_train/SOLAR_5epoch_sm04_v3_round1_argu_qlora_recall_top_100_rank/"
python -u get_doc_embeddings.py ${save_name} ${path_pre} ${model_path} ${lora_path} ${num}
python -u get_train_recall.py ${model_version} ${model_path}
python -u get_test_rank_result.py ${model_version} ${rank_model_path} ${num} ${rank_lora_path}



