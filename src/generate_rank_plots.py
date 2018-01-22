import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
for i in labels:
    
    medians[i] = {}
    fp = open("../data/" + dset + "/" + i +"_medians.out", "r")
    for line in fp:
        index = line.split("\t")[0]
        values = line.split("\t")[1].split("[")[1].split("]")[0].split(", ")
        medians[i][index] = values
    fp.close()
for key, value in medians.items():
    print(key)

for i in range(len(labels)):
    focus = labels[i]
    comp = labels[(i+1)%2]
   
     
    fp = open("../data/" + dset + "/" + focus +"_ranklist.out", "r")
    rankings = fp.readline().split("\n")[0].split(",")
    fp.close()
    fig = plt.figure(dpi=300, figsize=(14,9), tight_layout={'pad':3})
    ax = plt.subplot(111)
    ax.set_title("Feature Rankings for " + title[focus], fontsize=22)
    ax.set_ylabel("Cross Validated Rank", fontsize=22)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    
    for j in range(0,10):
        v = np.array(medians[focus][rankings[j]]).astype(np.float)
        x = np.random.normal(2*j+0.8, 0.04, len(v))
	plt.plot(x,v, mec='k', color='green', ms=4, marker="o", linestyle="None", zorder=1)
        dash_y = np.median(v)
        plt.plot((2*j+0.6, 2*j+1),(dash_y, dash_y), c="orange", lw=4, mec='k')

        v = np.array(medians[comp][rankings[j]]).astype(np.float)
        x = np.random.normal(2*j+1.2, 0.04, len(v))
        plt.plot(x,v,color='red', mec='k', ms=4, marker="o", linestyle="None", zorder=1)
        dash_y = np.median(v)
        plt.plot((2*j+1, 2*j+1.4),(dash_y, dash_y), c="orange", lw=4, mec='k')




    plt.xlim(0,21)
    plt.setp(ax, xticks=[2 * y + 1 for y in range(10)], xticklabels=[x for x in rankings])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_fontsize(22)
	    

    focus_patch = mpatches.Patch(color='green', label=title[focus])
    compare_patch = mpatches.Patch(color='red', label=title[comp])
    plt.legend(handles=[focus_patch, compare_patch], bbox_to_anchor=(1, 1), loc='lower right')
    #plt.show()
    plt.savefig('../images/' + dset + '_' + labels[i] + '_rankings.png')

