import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_venn import venn3
from parse_result import parse_result
from scipy import interp

plt.switch_backend('agg')

dset = sys.argv[1]

title = {}
title['n'] = "Healthy Individuals"
title['cirrhosis'] = "Individuals with Cirrhosis"
title['t2d'] = "Individuals with T2D"
title['leaness'] = "Lean Individuals"
title['obesity'] = "Obese Individuals"

fp = open("../data/" + dset +"/label_reference.txt", 'r')
labels = fp.readline().split("['")[1].split("']")[0].split("' '")
fp.close()
medians = {}

cnn_rank_list = []
rf_rank_list = []
lasso_rank_list = []

   
     
fp = pd.read_csv("../data/" + dset + "/tree_scores.out", sep="\t")
fp[['Score']] = fp[['Score']].apply(pd.to_numeric).apply(np.abs)
rankings = fp.sort_values(by=(['Score']), ascending=False)

fp = open("../data/" + dset + "/otu.csv", 'r')
otus = fp.readline().split(",")

i = 0
while len(cnn_rank_list) < 20:
    if rankings.iloc[i]['Node'] in otus:
	    cnn_rank_list.append(rankings.iloc[i]['Node'])
    i += 1    

cnn_rank_list = np.array(cnn_rank_list).reshape(20)
print(cnn_rank_list)
		
fp = pd.read_csv("../data/" + dset + "/" + "rf_features.txt", sep="\t", header=None)
rf_rank_list = np.array(fp[[0]].iloc[0:20]).reshape(20)
print(rf_rank_list)

fp = pd.read_csv("../data/" + dset + "/" + "lasso_features.txt", sep="\t", header=None)
lasso_rank_list = np.array(fp[[0]].iloc[0:20]).reshape(20)
print(lasso_rank_list)

set1 = set(cnn_rank_list)
set2 = set(rf_rank_list)
set3 = set(lasso_rank_list)

fig = plt.figure(dpi=300, figsize=(9,9), tight_layout={'pad':3})
ax = plt.subplot(111)
ax.set_title("Feature Overlap for " + dset, fontsize=18)
venn3([set1, set2, set3], ('CNN', 'RF', 'LASSO'))
plt.savefig('../images/' + dset + '_feature_overlap.png')
