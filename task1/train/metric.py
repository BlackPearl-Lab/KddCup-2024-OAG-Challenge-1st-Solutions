from sklearn import metrics
import numpy as np
# from .utils import *

def weighted_metric(pred_score:list , label: list) -> float:
    num_pred = [len(i) for i in pred_score]
    num_label = [len(i) for i in label]
    assert all(a==b for a, b in zip(num_pred, num_label))
    pred_label = []
    for i in pred_score:
        pred_label.append([1 if j>=0.5 else 0 for j in i])
    acc_pred = [metrics.accuracy_score(l,p) for l,p in zip(label,pred_label)]
    f1_pred = [metrics.f1_score(l,p) for l,p in zip(label,pred_label)]
    ap = [metrics.average_precision_score(l,p) for l,p in zip(label,pred_score)]
    auc = [metrics.roc_auc_score(l,p) for l,p in zip(label,pred_score)]
    # abnormal = 0, normal = 1

    profile_metric = list(zip(auc, ap, acc_pred, f1_pred, ))
    
    num0 = np.array([i.count(0) for i in label])
    weight = num0/np.array(num0.sum())
    mean_AUC = sum(weight * auc)
    mAP = sum(weight * ap)
    weighted_acc = sum(weight * acc_pred)
    weighted_f1 = sum(weight * f1_pred) 
    return mean_AUC,mAP,weighted_acc,weighted_f1,profile_metric

def compute_metric(ground_truth:dict, res: dict):
    
    pred_list = []
    label_list = []
    
    for author,pubs in ground_truth.items():
        sub_res = res[author]
        keys = pubs['normal_data'] +pubs['outliers']
        res_keys = list(res[author]['normal_data'].keys()) + list(res[author]['outliers'].keys())
        assert set(keys) == set(res_keys)

        label = [1]* len(pubs['normal_data'])+[0]* len(pubs['outliers'])
        
        pred = []
        for i in keys:
            if i in sub_res['normal_data'].keys() and i not in sub_res['outliers'].keys():
                pred.append(sub_res['normal_data'][i])
            elif i in sub_res['outliers'].keys() and i not in sub_res['normal_data'].keys():
                pred.append(sub_res['outliers'][i])
            else:
                #报错
                raise Exception('缺少预测值')

        pred_list.append(pred)
        label_list.append(label)
    mean_AUC,mAP,acc, f1,profile_metric = weighted_metric(pred_list, label_list)
    res = zip(ground_truth.keys(),profile_metric)
    import json
    with open('metric_pair.json','w') as f:
        json.dump(list(res),f)
    return mean_AUC,mAP,acc, f1
