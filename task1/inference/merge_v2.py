import json

with open(f'../test_result/title_v1.json', 'r') as f:
    data1 = json.load(f)
with open(f'../test_result/title_v2.json', 'r') as f:
    data2 = json.load(f)
with open(f'../test_result/title_v2_downsample.json', 'r') as f:
    data3 = json.load(f)

title_tta = {}
for author,score_dict in data1.items():
    title_tta[author] = {}
    for pid,score in score_dict.items():
        title_tta[author][pid] = data1[author][pid] * 0.3 + data2[author][pid] * 0.3 + data3[author][pid] * 0.4
        
        
with open(f'../test_result/author_v1.json', 'r') as f:
    data1 = json.load(f)
with open(f'../test_result/author_v2.json', 'r') as f:
    data2 = json.load(f)
with open(f'../test_result/author_v2_len_500.json', 'r') as f:
    data3 = json.load(f)

author_tta = {}
for author,score_dict in data1.items():
    author_tta[author] = {}
    for pid,score in score_dict.items():
        author_tta[author][pid] = data1[author][pid] * 0.3 + data2[author][pid] * 0.3 + data3[author][pid] * 0.4


with open(f'../test_result/all_info_v1.json', 'r') as f:
    data1 = json.load(f)
with open(f'../test_result/all_info_v2.json', 'r') as f:
    data2 = json.load(f)

all_info_tta = {}
for author,score_dict in data1.items():
    all_info_tta[author] = {}
    for pid,score in score_dict.items():
        all_info_tta[author][pid] = data1[author][pid] * 0.5 + data2[author][pid] * 0.5
        

merge_dict = {}
for author,score_dict in data1.items():
    merge_dict[author] = {}
    for pid,score in score_dict.items():
        merge_dict[author][pid] = title_tta[author][pid] * 0.35 + author_tta[author][pid] * 0.3 + all_info_tta[author][pid] * 0.35
        
with open(f'../test_result/final_submit.json', 'w') as f:
    json.dump(merge_dict, f)