import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, 
def parse_result (dset, model):
    f = open("../results_for_paper/" + dset + "/metaml/" + model + ".txt", "r")

    acc_list=[]
    roc_list=[]
    prec_list=[]
    recall_list=[]
    f1_list=[]
    tpr_list=[]
    fpr_list=[]
    thresh_list=[]

    f.readline()
    for i in range(0, 100):
        f.readline()
	y = np.array(f.readline().split("true labels\t")[1].split("\t"), dtype=np.int32)
	pred = np.array(f.readline().split("estimated labels\t")[1].split("\t"), dtype=np.int32)
	prob = np.array(f.readline().split("estimated probabilities\t")[1].split("\t"), dtype=np.float)
        acc_list.append(accuracy_score(y, pred))
        roc_list.append(roc_auc_score(y, prob))
        prec_list.append(precision_score(y, pred, average="weighted"))
        recall_list.append(recall_score(y, pred, average="weighted"))
        f1_listl.append(f1_score(y, pred, average="weighted"))
        fpr, tpr, thresh = roc_curve(y, probs)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        thresh_list.append(thresh)

    acc_list = np.array(acc_list).reshape((1,100))
    roc_list = np.array(roc_list).reshape((1,100))
    prec_list = np.array(prec_list).reshape((1,100))
    recall_list = np.array(recall_list).reshape((1,100))
    f1_list = np.array(f1_list).reshape((1,100))
    f.close()

    return acc_list, roc_list, prec_list, recall_list, f1_list, tpr_list, fpr_list, thresh_list

