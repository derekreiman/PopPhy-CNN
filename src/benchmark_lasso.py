# Third-party libraries
import numpy as np
import os
import sys
import struct
from array import array as pyarray
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

if __name__ == '__main__':

	rf_accuracy = []
	rf_roc_auc = []
	rf_precision = []
	rf_recall = []
	rf_f_score = []
	rf_pred = []
	rf_prob = []
	data = sys.argv[1]
	norm = sys.argv[2]

	fi = {}
	fp = open("../data/" + str(data) + "/otu.csv", 'r')
	features = fp.readline().split(",")
	fp.close()

	for f in features:
		fi[f]=0
	
	
	for set in range(0,10):
		for cv in range(0,10):
			print(set,cv)
			dir = "../data/" + str(data) + "/data_sets/" + norm +"/CV_" + str(set) + "/" + str(cv)
			x = np.loadtxt(dir+'/benchmark_train_data.csv', delimiter=',')
			y = np.loadtxt(dir+'/benchmark_train_labels.csv', delimiter=',')
			tx = np.loadtxt(dir+'/benchmark_test_data.csv', delimiter=',')
			ty = np.loadtxt(dir+'/benchmark_test_labels.csv', delimiter=',')
			clf = LassoCV(alphas=np.logspace(-4, -0.5, 50), cv=5, n_jobs=-1)
			clf.fit(x, y)
			prob = [row for row in clf.predict(tx)]
			pred = [int(i > 0.5) for i in prob]
			rf_accuracy.append(clf.score(tx,ty))
			rf_roc_auc.append(roc_auc_score(ty, prob))
			rf_precision.append(precision_score(ty, pred, average='weighted'))
			rf_recall.append(recall_score(ty, pred, average='weighted'))
			rf_f_score.append(f1_score(ty, pred, average='weighted'))
			rf_pred.append(pred)
			rf_prob.append(prob)
			i=0
			for f in features:
				fi[f] += abs(clf.coef_[i])/sum(abs(clf.coef_))
				i += 1
			
	for f in features:
		fi[f] = fi[f]/100

	fp = open("../data/" + str(data) + "/lasso_features.txt", 'w') 
	for key, value in sorted(fi.iteritems(), key=lambda(k,v):(v,k), reverse=True):
	  fp.write(str(key) + "\t" + str(value) + "\n")
	fp.close()


	print("Accuracy = " + str(np.mean(rf_accuracy)) + " (" + str(np.std(rf_accuracy)) + ")\n")
	print(rf_accuracy)
	print("\n\nROC AUC = " + str(np.mean(rf_roc_auc)) + " (" + str(np.std(rf_roc_auc)) + ")\n")
	print(rf_roc_auc)
	print("\n\nPrecision = " + str(np.mean(rf_precision)) + " (" + str(np.std(rf_precision)) + ")\n")
	print("Recall = " + str(np.mean(rf_recall)) + " (" + str(np.std(rf_recall)) + ")\n")
	print("F1 = " + str(np.mean(rf_f_score)) + " (" + str(np.std(rf_f_score)) + ")\n")

	dir = "../results_for_paper/" + data + "/" + norm
	f = open(dir + "/lasso.txt", 'w')
	f.write("Mean Accuracy: " + str(np.mean(rf_accuracy)) + " (" + str(np.std(rf_accuracy))+ ")\n")
	f.write(str(rf_accuracy) + "\n")
	f.write("\nMean ROC: " + str(np.mean(rf_roc_auc)) + " (" + str(np.std(rf_roc_auc))+ ")\n")
	f.write(str(rf_roc_auc) + "\n")
	f.write("\nMean Precision: " + str(np.mean(rf_precision)) + " (" + str(np.std(rf_precision))+ ")\n")
	f.write(str(rf_precision) + "\n")
	f.write("\nMean Recall: " + str(np.mean(rf_recall)) + " (" + str(np.std(rf_recall))+ ")\n")
	f.write(str(rf_recall) + "\n")
	f.write("\nMean F-score: " + str(np.mean(rf_f_score)) + " (" + str(np.std(rf_f_score))+ ")\n")
	f.write(str(rf_f_score) + "\n")

	for i in range(0,100):
		f.write("\nPredictions for " + str(i) + "\n")
		f.write("\n" + str(rf_pred[i]) + "\n")
		f.write("\n" + str(rf_prob[i]) + "\n")
	f.close()
		
