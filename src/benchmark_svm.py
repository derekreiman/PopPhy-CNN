import sys
import numpy as np
import os
import struct
from array import array as pyarray
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from sklearn.grid_search import GridSearchCV
import pandas as pd

if __name__ == '__main__':
	svm_accuracy = []
	svm_roc_auc = []
	svm_precision = []
	svm_recall = []
	svm_f_score = []
	svm_pred = []
	svm_prob = []
	data = sys.argv[1]
	norm = sys.argv[2]

	fp = open("../data/" + data +"/label_reference.txt", 'r')
	labels = fp.readline().split("['")[1].split("']")[0].split("' '")
	fp.close()

	for set in range(0,10):
		for cv in range(0,10):
			print(set,cv)
			dir = "../data/" + str(data) + "/data_sets/" + norm + "/CV_" + str(set) + "/" + str(cv)
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
			svm_accuracy.append(clf.score(tx,ty))
			svm_roc_auc.append(roc_auc_score(ty, prob))
			svm_precision.append(precision_score(ty, pred, average='weighted'))
			svm_recall.append(recall_score(ty, pred, average='weighted'))
			svm_f_score.append(f1_score(ty, pred, average='weighted'))
			svm_pred.append(pred)
			svm_prob.append(prob)

	print("Accuracy = " + str(np.mean(svm_accuracy)) + " (" + str(np.std(svm_accuracy)) + ")\n")
	print(svm_accuracy)
	print("\n\nROC AUC = " + str(np.mean(svm_roc_auc)) + " (" + str(np.std(svm_roc_auc)) + ")\n")
	print(svm_roc_auc)
	print("\n\nPrecision = " + str(np.mean(svm_precision)) + " (" + str(np.std(svm_precision)) + ")\n")
	print("Recall = " + str(np.mean(svm_recall)) + " (" + str(np.std(svm_recall)) + ")\n")
	print("F1 = " + str(np.mean(svm_f_score)) + " (" + str(np.std(svm_f_score)) + ")\n")


	dir = "../results_for_paper/" + data + "/" + norm
	f = open(dir + "/svm.txt", 'w')
	f.write("Mean Accuracy: " + str(np.mean(svm_accuracy)) + " (" + str(np.std(svm_accuracy))+ ")\n")
	f.write(str(svm_accuracy) + "\n")
	f.write("\nMean ROC: " + str(np.mean(svm_roc_auc)) + " (" + str(np.std(svm_roc_auc))+ ")\n")
	f.write(str(svm_roc_auc) + "\n")
	f.write("\nMean Precision: " + str(np.mean(svm_precision)) + " (" + str(np.std(svm_precision))+ ")\n")
	f.write(str(svm_precision) + "\n")
	f.write("\nMean Recall: " + str(np.mean(svm_recall)) + " (" + str(np.std(svm_recall))+ ")\n")
	f.write(str(svm_recall) + "\n")
	f.write("\nMean F-score: " + str(np.mean(svm_f_score)) + " (" + str(np.std(svm_f_score))+ ")\n")
	f.write(str(svm_f_score) + "\n")

	for i in range(0,100):
		f.write("\nPredictions for " + str(i) + "\n")
		f.write("\n" + str(svm_pred[i]) + "\n")
		f.write("\n" + str(svm_prob[i]) + "\n")
	f.close()   

