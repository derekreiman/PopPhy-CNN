import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def parse_result(dset, model):
    f = open("../results_for_paper/" + dset + "/raw_noscale/" + model + ".txt", "r")

    f.readline()
    acc_list = f.readline().split("[")[1].split("]")[0].split(",")
    acc_list = map(float, acc_list)
    f.readline()
    f.readline()
    roc_list = f.readline().split("[")[1].split("]")[0].split(",")
    roc_list = map(float, roc_list)
    f.readline()
    f.readline()
    prec_list = f.readline().split("[")[1].split("]")[0].split(",")
    prec_list = map(float, prec_list)
    f.readline()
    f.readline()
    recall_list = f.readline().split("[")[1].split("]")[0].split(",")
    recall_list = map(float, recall_list)
    f.readline()
    f.readline()
    f1_list = f.readline().split("[")[1].split("]")[0].split(",")
    f1_list = map(float, f1_list)

    tpr_list = []
    fpr_list = []
    thresh_list = []
    if model == "lasso":
        acc_list = []

    for set in range(0,10):
        for cv in range(0,10):
          f.readline()
          f.readline()
          f.readline()
          f.readline()
          y = np.loadtxt("../data/" + dset + "/data_sets/raw_noscale/CV_" + str(set) + "/" + str(cv) + "/benchmark_test_labels.csv", dtype=np.int32)
          f.readline()    
          probs = f.readline().split("[")[1].split("]")[0].split(",")
	  print(probs)
	  probs = [float(i) for i in probs]
          if model == "lasso":
              pred = [int(i > 0.5) for i in probs]
          fpr, tpr, thresh = roc_curve(y, probs)
          tpr_list.append(tpr)
          fpr_list.append(fpr)
          thresh_list.append(thresh)

    f.close()

    return acc_list, roc_list, prec_list, recall_list, f1_list, tpr_list, fpr_list, thresh_list



def parse_result_metaml(dset, model):
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
        y = np.array(f.readline().split("true labels\t")[1].split("\t\n")[0].split("\t"), dtype=np.int32)
        pred = np.array(f.readline().split("estimated labels\t")[1].split("\t\n")[0].split("\t"), dtype=np.int32)
        prob = np.array(f.readline().split("estimated probabilities\t")[1].split("\t\n")[0].split("\t"), dtype=np.float)
        acc_list.append(accuracy_score(y, pred))
        roc_list.append(roc_auc_score(y, prob))
        prec_list.append(precision_score(y, pred, average="weighted"))
        recall_list.append(recall_score(y, pred, average="weighted"))
        f1_list.append(f1_score(y, pred, average="weighted"))
        fpr, tpr, thresh = roc_curve(y, prob)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        thresh_list.append(thresh)
        f.readline()

    acc_list = np.array(acc_list).reshape((1,100))
    roc_list = np.array(roc_list).reshape((1,100))
    prec_list = np.array(prec_list).reshape((1,100))
    recall_list = np.array(recall_list).reshape((1,100))
    f1_list = np.array(f1_list).reshape((1,100))
    f.close()

    return acc_list, roc_list, prec_list, recall_list, f1_list, tpr_list, fpr_list, thresh_list

