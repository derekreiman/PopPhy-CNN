import numpy as np
import pandas as pd
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
parser.add_argument("-m", "--method", default="CV", help="CV or Holdout method.") #Holdout method TBI
parser.add_argument("-d", "--dataset", default="Cirrhosis", 	help="Name of dataset in data folder.")
parser.add_argument("-n", "--splits", default=10, type=int, help="Number of cross validated splits.")
parser.add_argument("-s", "--sets", default=10, type=int, help="Number of datasets to generate")
parser.add_argument("-c", "--classifier", default="RF", help="Classifier to use.")

args = parser.parse_args()

accuracy = []
roc_auc = []
precision = []
recall = []
f_score = []
pred_list = []
prob_list = []
method = args.method
total = args.splits * args.sets
data = args.dataset
classifier = args.classifier

fi = {}
fp = open("../data/" + str(data) + "/otu.csv", 'r')
features = fp.readline().split(",")
fp.close()

for f in features:
	fi[f]=0

for set in range(0,args.sets):
	for cv in range(0,args.splits):
		dir = "../data/" + str(data) + "/data_sets/CV_" + str(set) + "/" + str(cv)
	
		if classifier=="RF":
			x = np.loadtxt(dir+'/benchmark_train_data.csv', delimiter=',')
			y = np.loadtxt(dir+'/benchmark_train_labels.csv', delimiter=',')
			tx = np.loadtxt(dir+'/benchmark_test_data.csv', delimiter=',')
			ty = np.loadtxt(dir+'/benchmark_test_labels.csv', delimiter=',')
			clf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, n_jobs=-1)
			clf.fit(x,y)
			prob = [row[1] for row in clf.predict_proba(tx)]
			pred = [row for row in clf.predict(tx)]

		if classifier=="SVM":
                        x = pd.read_csv(dir+"/benchmark_train_data.csv", header=None, dtype=np.float64)
                        y = pd.DataFrame(np.loadtxt(dir+'/benchmark_train_labels.csv', delimiter=','))
                        tx = pd.read_csv(dir+"/benchmark_test_data.csv", header=None, dtype=np.float64)
                        ty = pd.DataFrame(np.loadtxt(dir+'/benchmark_test_labels.csv', delimiter=','))
			x = (x - x.min())/(x.max() - x.min())
			tx = (tx - tx.min())/(tx.max() - tx.min())
			x = x.fillna(value=0)
			tx = tx.fillna(value=0)
			cv_grid = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']}, {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']} ]
			clf = GridSearchCV(SVC(C=1, probability=True), param_grid=cv_grid, cv=StratifiedKFold(y[0], n_folds=5, shuffle=True), n_jobs=-1, scoring="accuracy")
			clf.fit(x, y[0])
			prob = [row[1] for row in clf.predict_proba(tx)]
			pred = [row for row in clf.predict(tx)]

		if classifier=="LASSO":
                        x = np.loadtxt(dir+'/benchmark_train_data.csv', delimiter=',')
                        y = np.loadtxt(dir+'/benchmark_train_labels.csv', delimiter=',')
                        tx = np.loadtxt(dir+'/benchmark_test_data.csv', delimiter=',')
                        ty = np.loadtxt(dir+'/benchmark_test_labels.csv', delimiter=',')
			clf = LassoCV(alphas=np.logspace(-4, -0.5, 50), cv=5, n_jobs=-1)
			clf.fit(x, y)
			prob = [row for row in clf.predict(tx)]
			pred = [int(i > 0.5) for i in prob]
			
		accuracy.append(clf.score(tx,ty))
		roc_auc.append(roc_auc_score(ty, prob))
		precision.append(precision_score(ty, pred, average='weighted'))
		recall.append(recall_score(ty, pred, average='weighted'))
		f_score.append(f1_score(ty, pred, average='weighted'))
		pred_list.append(pred)
		prob_list.append(prob)

		if classifier == "RF":
			i=0
			for f in features:
				fi[f] += clf.feature_importances_[i]
				i += 1

		if classifier == "LASSO":
			i=0
			for f in features:
				fi[f] += abs(clf.coef_[i])/sum(abs(clf.coef_))
				i += 1

if classifier == "LASSO" or classifier == "RF":				
	for f in features:
		fi[f] = fi[f]/total

	fp = open("../data/" + str(data) + "/" + classifier + "_features.txt", 'w') 
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

dir = "../data/" + str(data)
f = open(dir + "/data_sets/" + method + "_results_" + classifier + ".txt", 'w')
f.write("Mean Accuracy: " + str(np.mean(accuracy)) + " (" + str(np.std(accuracy))+ ")\n")
f.write(str(accuracy) + "\n")
f.write("\nMean ROC: " + str(np.mean(roc_auc)) + " (" + str(np.std(roc_auc))+ ")\n")
f.write(str(roc_auc) + "\n")
f.write("\nMean Precision: " + str(np.mean(precision)) + " (" + str(np.std(precision))+ ")\n")
f.write(str(precision) + "\n")
f.write("\nMean Recall: " + str(np.mean(recall)) + " (" + str(np.std(recall))+ ")\n")
f.write(str(recall) + "\n")
f.write("\nMean F-score: " + str(np.mean(f_score)) + " (" + str(np.std(f_score))+ ")\n")
f.write(str(f_score) + "\n")

for i in range(0,total):
	f.write("\nPredictions for " + str(i) + "\n")
	f.write("\n" + str(pred_list[i]) + "\n")
	f.write("\n" + str(prob_list[i]) + "\n")
f.close()
  
