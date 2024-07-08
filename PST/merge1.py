import pandas as pd
import json

all = ['addbib_addcite_step1950', 'addbib_addcite_step950']
part = [0.4, 0.6]
save = []
new_dict = {}

for path in all:
    with open(f'./eval_result/{path}.json', 'r') as f:
        pred = json.load(f)

    test = pd.read_pickle('./data/llm_final_title_addbib_moreinfo_processtext.pickle')
    test['pred'] = [pred[str(x)] for x in range(len(test))]
    test['pred'] = test['pred'].apply(lambda x : min(1.0, max(0.0, x)))

    with open('./data/PST/submission_example_test.json', 'r') as f:
        sub = json.load(f)

    new_sub = {}
    for k, v in sub.items():
        cut = test[test['idx'] == k].reset_index(drop=True)
        p = cut['pred'].tolist()
        new_sub[k] = p
    save.append(new_sub)

for k, v in save[0].items():
    tmp = []
    for j in range(len(save)):
        tmp.append([save[j][k][i] * part[j] for i in range(len(save[0][k]))])
    final = []
    for i in range(len(tmp[0])):
        sums = 0
        for j in range(len(save)):
            sums += tmp[j][i]
        final.append(sums)
    new_dict[k] = final
with open('./data/submission_final_allmerge.json', 'w') as f:
    json.dump(new_dict, f, indent=4, ensure_ascii=False)
