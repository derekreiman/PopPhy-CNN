# Third-party libraries
import numpy as np
import os
import sys
import struct
import argparse
from array import array as pyarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


parser = argparse.ArgumentParser(description="Prepare Data")
parser.add_argument("-m", "--method", default="CV", 	help="CV or Holdout method.") #Holdout method TBI
parser.add_argument("-d", "--dataset", default="Cirrhosis", 	help="Name of dataset in data folder.")
parser.add_argument("-n", "--splits", default=10, type=int, help="Number of cross validated splits.")
parser.add_argument("-s", "--sets", default=10, type=int, help="Number of datasets to generate")

args = parser.parse_args()

accuracy = []
roc_auc = []
precision = []
recall = []
f_score = []
pred = []
prob = []
method = args.method
total = args.splits * args.sets

fi = {}
fp = open("../data/" + str(data) + "/otu.csv", 'r')
features = fp.readline().split(",")
fp.close()

for f in features:
	fi[f]=0

for set in range(0,args.sets):
    for cv in range(0,args.splits):
        dir = "../data/" + str(data) + "/data_sets/" + "/CV_" + str(set) + "/" + str(cv)
        x = np.loadtxt(dir+'/benchmark_train_data.csv', delimiter=',')
        y = np.loadtxt(dir+'/benchmark_train_labels.csv', delimiter=',')
        tx = np.loadtxt(dir+'/benchmark_test_data.csv', delimiter=',')
        ty = np.loadtxt(dir+'/benchmark_test_labels.csv', delimiter=',')
		
		if method=="RF":
			clf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, n_jobs=-1)
			prob = [row[1] for row in clf.predict_proba(tx)]
			pred = [row for row in clf.predict(tx)]
			
		if method=="SVM":
			x = (x - x.min())/(x.max() - x.min())
			tx = (tx - tx.min())/(tx.max() - tx.min())
			x = x.fillna(value=0)
			tx = tx.fillna(value=0)
			cv_grid = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']}, {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']} ]
			clf = GridSearchCV(SVC(C=1, probability=True), param_grid=cv_grid, cv=StratifiedKFold(y, n_folds=5, shuffle=True), n_jobs=-1, scoring="accuracy")
			prob = [row[1] for row in clf.predict_proba(tx)]
			pred = [row for row in clf.predict(tx)]

		if method=="LASSO":
			clf = LassoCV(alphas=np.logspace(-4, -0.5, 50), cv=5, n_jobs=-1)
			clf.fit(x, y)
			prob = [row for row in clf.predict(tx)]
			pred = [int(i > 0.5) for i in prob]
			
        accuracy.append(clf.score(tx,ty))
        roc_auc.append(roc_auc_score(ty, prob))
        precision.append(precision_score(ty, pred, average='weighted'))
        recall.append(recall_score(ty, pred, average='weighted'))
        f_score.append(f1_score(ty, pred, average='weighted'))
        pred.append(pred)
        prob.append(prob)
		
		if method == "RF":
			i=0
			for f in features:
				fi[f] += clf.feature_importances_[i]
				i += 1

		if method == "LASSO":
			i=0
			for f in features:
				fi[f] += abs(clf.coef_[i])/sum(abs(clf.coef_))
	
	i += 1
if method == "LASSO" or method == "RF":				
	for f in features:
		fi[f] = fi[f]/total

	fp = open("../data/" + str(data) + "/" + method + "_features.txt", 'w') 
	for key, value in sorted(fi.iteritems(), key=lambda(k,v):(v,k), reverse=True):
	  fp.write(str(key) + "\t" + str(value) + "\n")
	fp.close()

print("Accuracy = " + str(np.mean(accuracy)) + " (" + str(np.std(accuracy)) + ")\n")
print(accuracy)
print("\n\nROC AUC = " + str(np.mean(roc_auc)) + " (" + str(np.std(roc_auc)) + ")\n")
print(roc_auc)
print("\n\nPrecision = " + str(np.mean(precision)) + " (" + str(np.std(precision)) + ")\n")
print("Recall = " + str(np.mean(recall)) + " (" + str(np.std(recall)) + ")\n")
print("F1 = " + str(np.mean(f_score)) + " (" + str(np.std(f_score)) + ")\n")

dir = "../results_for_paper/" + data
f = open(dir + "/" + method + ".txt", 'w')
f.write("Mean Accuracy: " + str(np.mean(accuracy)) + " (" + str(np.std(accuracy))+ ")\n")
f.write(str(rf_accuracy) + "\n")
f.write("\nMean ROC: " + str(np.mean(roc_auc)) + " (" + str(np.std(roc_auc))+ ")\n")
f.write(str(rf_roc_auc) + "\n")
f.write("\nMean Precision: " + str(np.mean(precision)) + " (" + str(np.std(precision))+ ")\n")
f.write(str(rf_precision) + "\n")
f.write("\nMean Recall: " + str(np.mean(recall)) + " (" + str(np.std(recall))+ ")\n")
f.write(str(rf_recall) + "\n")
f.write("\nMean F-score: " + str(np.mean(f_score)) + " (" + str(np.std(f_score))+ ")\n")
f.write(str(rf_f_score) + "\n")

for i in range(0,total):
    f.write("\nPredictions for " + str(i) + "\n")
    f.write("\n" + str(pred[i]) + "\n")
    f.write("\n" + str(prob[i]) + "\n")
f.close()
  
