import sys
import numpy as np
import pandas as pd
from array import array as pyarray
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import warnings

import argparse

parser = argparse.ArgumentParser(description="PopPhy-CNN Training")
parser.add_argument("-d", "--dataset", default="Cirrhosis",     help="Name of dataset in data folder.")
args = parser.parse_args()

dset = args.dataset

if __name__ == "__main__":

	warnings.filterwarnings("ignore")

	fp = open("../data/" + dset +"/label_reference.txt", 'r')
	labels = fp.readline().split("['")[1].split("']")[0].split("' '")
	fp.close()

	df = pd.read_csv("../data/" + dset + "/count_matrix.csv", header=None)

	fp = open("../data/" + dset + "/otu.csv", 'r')
	col = fp.readline().split(",")
	if col[-1] == '':
		col = col[0:-1]
	fp.close()
	df.columns=col
	ranklist = {}

	for i in range(2):
		fp = open("../data/" + dset + "/" + labels[i] + "_ranklist.out", 'r')
		ranklist[i] = fp.readline().split(",")
		fp.close()

	fp = pd.read_csv("../data/" + dset + "/snr_features.txt", header=None, sep="\t")
	snr_features = np.array(fp[[0]]).reshape(len(col))
				
	fp = pd.read_csv("../data/" + dset + "/rf_features.txt", header=None, sep="\t")
	rf_features = np.array(fp[[0]]).reshape(len(col))

        fp = pd.read_csv("../data/" + dset + "/wilcox.csv", header=0, sep=",")
        wilcox_features = np.array(fp["OTU"]).reshape(len(col))


	fp = pd.read_csv("../data/" + dset + "/tree_scores.out", sep="\t")
	fp[['Score']] = fp[['Score']].apply(pd.to_numeric).apply(np.abs)
	rankings = fp.sort_values(by=(['Score']), ascending=False)

	fp = open("../data/" + dset + "/features_svm.txt", "w")

	auc_list = {}
	for feat in range(1,26, 4):
		fp.write(str(feat) + ",")
		num_features = feat
		for i in range(0,6):
			svm_roc_auc = []

			if i == 0 or i == 1:
				print("CNN Single")
				n = 0
				features = []
				while len(features) < num_features:
					if ranklist[i][n] in df.columns:
						features.append(ranklist[i][n])
					n += 1
						
			elif i == 3:
				print("SNR")
				features = []
				n = 0
				while len(features) < num_features:
					features.append(snr_features[n])
					n += 1
                        elif i == 4:
				print("RF")
                                features = []
                                n = 0
                                while len(features) < num_features:
                                        features.append(rf_features[n])
                                        n += 1

			elif i == 2:
				print("CNN Join")
				features = []
				n = 0
				while len(features) < num_features:
					if rankings.iloc[n]['Node'] in df.columns:
						features.append(rankings.iloc[n]['Node'])
					n += 1
			elif i == 5:
				print("Wilcox")
				features = []
				n = 0
				while len(features) < num_features:
					features.append(wilcox_features[n])
					n += 1



			print(features)
			x = df[features]
			x = (x-x.min())/(x.max() - x.min())
			y = pd.factorize(np.loadtxt("../data/" + dset + "/labels.txt", dtype=np.str_))[0]
			y = pd.DataFrame(y)
			cv_grid = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']}, {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']} ]

			scores = []
			rocs = []
			tys = []
			count = 0
			print("Running....")
			for j in range(10):
				count = 0
				k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=j)
				for train, test in k_fold.split(x, y[0]):
					clf = GridSearchCV(SVC(C=1, probability=True), param_grid=cv_grid, cv=5, n_jobs=-1, scoring="accuracy")
					clf.fit(x.loc[train].values, y.loc[train].values.flatten().astype('int'))
					tx = x.loc[test]
					ty = y.loc[test]
					prob = [row[1] for row in clf.predict_proba(tx)]
					pred = clf.predict(tx)
					tys.append(ty)
					svm_roc_auc.append(roc_auc_score(ty, prob))
			fp.write(str(np.mean(svm_roc_auc)) + ",")
			print(str(np.mean(svm_roc_auc)))
		fp.write("\n")
	fp.close()
