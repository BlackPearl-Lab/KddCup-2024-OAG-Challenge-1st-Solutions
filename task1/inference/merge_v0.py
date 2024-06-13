# import json

# with open(f'../test_result/title_v0_seed42.json', 'r') as f:
#     data1 = json.load(f)
# with open(f'../test_result/title_v0_seed3407.json', 'r') as f:
#     data2 = json.load(f)
# title_tta = {}
# for author,score_dict in data1.items():
#     title_tta[author] = {}
#     for pid,score in score_dict.items():
#         title_tta[author][pid] = data1[author][pid] * 0.5 + data2[author][pid] * 0.5
        

# with open(f'../test_result/author_v0_seed42.json', 'r') as f:
#     data3 = json.load(f)
# with open(f'../test_result/testb_baseline_org_add_gcn.pkl.json', 'r') as f:
#     data4 = json.load(f)
      
# merged_dict = {}
# for author,score_dict in data1.items():
#     merged_dict[author] = {}
#     for pid,score in score_dict.items():
#         merged_dict[author][pid] = data3[author][pid] * 0.4 + title_tta[author][pid] * 0.3 + data4[author][pid] * 0.3
        
# with open(f'../test_result/merge_title_author_gcn.json', 'w') as f:
#     json.dump(merged_dict, f)

import json

with open(f'../test_result/title_v0_seed42.json', 'r') as f:
    data1 = json.load(f)
with open(f'../test_result/title_v0_seed3407.json', 'r') as f:
    data2 = json.load(f)
title_tta = {}
for author,score_dict in data1.items():
    title_tta[author] = {}
    for pid,score in score_dict.items():
        title_tta[author][pid] = data1[author][pid] * 0.5 + data2[author][pid] * 0.5
        

with open(f'../test_result/author_v0_seed42.json', 'r') as f:
    data3 = json.load(f)
      
merged_dict = {}
for author,score_dict in data1.items():
    merged_dict[author] = {}
    for pid,score in score_dict.items():
        merged_dict[author][pid] = data3[author][pid] * 0.5 + title_tta[author][pid] * 0.5
        
with open(f'../test_result/merge_title_author.json', 'w') as f:
    json.dump(merged_dict, f)