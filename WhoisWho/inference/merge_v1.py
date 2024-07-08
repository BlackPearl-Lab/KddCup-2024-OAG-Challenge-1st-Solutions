import json

with open(f'../test_result/title_v1.json', 'r') as f:
    data1 = json.load(f)
with open(f'../test_result/author_v1.json', 'r') as f:
    data2 = json.load(f)
with open(f'../test_result/all_info_v1.json', 'r') as f:
    data3 = json.load(f)

merged_dict = {}
for author,score_dict in data1.items():
    merged_dict[author] = {}
    for pid,score in score_dict.items():
        merged_dict[author][pid] = data1[author][pid] * 0.3 + data2[author][pid] * 0.3 + data3[author][pid] * 0.4
        
with open(f'../test_result/merge_all_334.json', 'w') as f:
    json.dump(merged_dict, f)