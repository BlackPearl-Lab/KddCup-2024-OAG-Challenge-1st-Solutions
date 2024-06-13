
import glob
import json
datas = []
for path in glob.glob("../sub_test/*rank*.txt"):
    print(path)
    f = open(path, 'r')
    data = []
    for line in f.readlines():
        data.append(line.strip('\n').split(','))
        if "new_1" in path or "new_2" in path or "new_3" in path:
            data[-1] = data[-1][:20]
    datas.append(data)

for data in datas:
    print(len(data[0]))
    print(len(data))

final_res = []
for cnt in range(len(datas[0])):
    ps = []
    for k in range(len(datas)):
        # print(k)
        # # print(len(datas[k][cnt]))
        # if k==3:
        #     print(datas[k][cnt])
        ps.append({pid: 1 / (i + 1) for i, pid in enumerate(datas[k][cnt])})
    res = {}
    for p_dict in ps:
        for p, i in p_dict.items():
            res[p] = res.get(p, 1 / 21) + i
    res = sorted(res.items(), key=lambda x: -x[1])
    res = [a for a, b in res[:20]]
    final_res.append(res)

f = open(f"../sub_test/merge_7_model_last.txt", 'w')
for pid in final_res:
    pid = ",".join(pid[:20])
    f.write(pid + "\n")
f.close()
