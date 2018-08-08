import sys
import pandas as pd
import numpy as np
import argparse
from popphy_io import load_network, load_data_from_file
from theano.tensor.nnet import conv2d
from graph import Graph, Node

parser = argparse.ArgumentParser(description="PopPhy-CNN Feature Extraction")
parser.add_argument("-n", "--splits", default=10, type=int, help="Number of cross validated splits.")
parser.add_argument("-s", "--sets", default=10, type=int, help="Number of datasets")
parser.add_argument("-d", "--dataset", default="Cirrhosis",     help="Name of dataset in data folder.")
parser.add_argument("-m", "--method", default="CV", help="CV or holdout.")
args = parser.parse_args()

dset = args.dataset
method = args.method

if method == "CV":
    prefix = "CV_"
    num_splits = int(args.splits)

if method == "holdout" or args.method == "HO":
    prefix = "HO_"
    num_splits=1

num_sets = int(args.sets)

#Get reference graph            
g = Graph()
g.build_graph("../data/" + dset + "/newick.txt")
ref = g.get_ref()

ref_val = np.zeros((ref.shape[0], ref.shape[1]))
num_nodes = g.get_node_count()

rankings = {}
scores = {}
node_names = g.get_dictionary()

fp = open("../data/" + dset +"/label_reference.txt", 'r')
labels = fp.readline().split("['")[1].split("']")[0].split("' '")
fp.close()

fp = open("../data/" + dset +"/otu.csv", 'r')
otus = fp.readline().split(",")
fp.close()

for i in range(0, len(labels)):
	rankings[i] = {}
        scores[i] = {}
	for j in node_names:
	    rankings[i][j] = []
            scores[i][j] = []


for roc in range(0,num_sets * num_splits):
        set = str(roc / num_splits)
        cv = str(roc % num_splits)
	if method=="CV":
		net = load_network("../data/" + dset + "/data_sets/" + prefix + set + "/" + cv)
	if method=="HO" or method=="holdout":
		net = load_network("../data/" + dset + "/data_sets/" + prefix + set + "/" + cv)
	train, test, validation = load_data_from_file(dset, "CV", set, cv)

	num_classes = net.layers[-1].w.eval().shape[1]
	num_train = train[1].eval().shape[0]
	num_test = test[1].eval().shape[0]
	num_samp = num_train
	w = net.layers[0].w.eval()
	num_maps = w.shape[0]
	w_row = w.shape[2]
	w_col = w.shape[3]

	data_set = train
	data_shape = data_set[0].eval().shape

	fm = {}
	data = {}
	for i in range(0, num_classes):
	    fm[i] = []
	    data[i] = []

	for i in range(0, num_samp):
	    f = conv2d(data_set[0][i].reshape([1,1,data_shape[1], data_shape[2]]), w)
	    y = data_set[1][i].eval()
	    fm[int(y)].append(f.eval())
	    data[int(y)].append(data_set[0][i].eval())

	for i in range(0, num_classes):
	    fm[i] = np.array(fm[i])
	    data[i] = np.array(data[i])

	fm_rows = fm[0][0].shape[2]
	fm_cols = fm[0][0].shape[3]

	theta1 = 0.3 #0.5	0.69
	theta2 = 0.3 #0\0.8

	#Get the top X max indices for each class and each feature map
	max_list = np.zeros((num_classes, num_maps, fm_rows * fm_cols))

	for i in range(0, num_classes):
	    for j in range(0, len(fm[i])):
		for k in range(0, num_maps):
		    maximums = np.argsort(fm[i][j][0][k].flatten())[::-1]
		    for l in range(0, int(round(theta1 * num_nodes))):
		        max_list[i][k][maximums[l]] += 1


	d = {"OTU":otus,"Max Score":np.zeros(len(otus)), "Cumulative Score":np.zeros(len(otus))}
	df = pd.DataFrame(data = d)
	results = {}

	for i in range(0, num_classes):
	    results[i] = df.set_index("OTU")
	#For each class
	for i in range(0, num_classes):
	    
	   #For each feature map...
	    for j in range(0, num_maps):
		
		#Order the feature map's maximums
		loc_list = max_list[i][j].argsort()[::-1]    
		
		#Store kernel weights
		w = np.rot90(np.rot90(net.layers[0].w.container.data[j][0]))
	    
		#For the top X maximums...
		for k in range(0, len(loc_list)):
		    
		    #Find the row and column location and isolate reference window
		    loc = loc_list[k]    
		    if max_list[i][j][loc] > int(round(len(fm[i]) * theta2)):
		        row = loc / fm_cols
		        col = loc % fm_cols
		        ref_window = ref[row:row + w_row, col:col + w_col]                
		        count = np.zeros((w_row,w_col))
		        
		        #Calculate the proportion of the contribution of each pixel to the convolution with the absolute value of weights
		        for l in range(0,len(fm[i])):
		            window =data[i][l][row:row + w_row, col:col + w_col]
		            abs_v = (abs(w) * window).sum()
		            v = (w * window)
		            for m in range(0, v.shape[0]):
		                for n in range(0, v.shape[1]):
		                    count[m,n] += v[m,n] / abs_v
		        
		        #Divide by number of samples
		        count = count/len(fm[i])
	      
		        #Print out features with a high enough value    
		        for m in range(0, w_row):
		            for n in range(0, w_col):
		                if count[m,n] > 0:
		                        if ref_window[m,n] in results[i].index:
		                            if count[m,n] > results[i].loc[ref_window[m,n], "Max Score"]:
		                                results[i].loc[ref_window[m,n], "Max Score"] = count[m,n]
		                            results[i].loc[ref_window[m,n], "Cumulative Score"] += count[m,n]
		                        else:
		                            results[i].loc[ref_window[m,n], "Max Score"] = count[m,n]    
		                            results[i].loc[ref_window[m,n], "Cumulative Score"] = count[m,n]    

	diff = {}

	for i in range(0, num_classes):
	    diff[i] = df.set_index("OTU")
	    for j in results[i].index:
	       for k in range(0, num_classes):
		   if i != k:
		       if j in results[k].index:
		           diff[i].loc[j, "Max Score"] = results[i].loc[j, "Max Score"] - results[k].loc[j, "Max Score"]
		           diff[i].loc[j, "Cumulative Score"] = results[i].loc[j, "Cumulative Score"] - results[k].loc[j, "Cumulative Score"]
		       else:
		           diff[i].loc[j, "Max Score"] = results[i].loc[j, "Max Score"]     
		           diff[i].loc[j, "Cumulative Score"] = results[i].loc[j, "Cumulative Score"]     

            rank = diff[i]["Max Score"].rank(ascending=False)
 	    for j in node_names:
                if j in rank.index:
		    rankings[i][j].append(rank.loc[j])
		    scores[i][j].append(diff[i].loc[j,"Max Score"])
	        else:
                    rankings[i][j].append(rank.shape[0] + 1)
                    scores[i][j].append(0)

medians = {}

for i in range(0, num_classes):
    medians[i] = {}
    for j in rankings[i]:
        medians[i][j] = np.median(rankings[i][j])
    
    f = open("../data/" + dset + "/" + labels[i] + "_ranklist.out", "w")
    for m in sorted(medians[i], key=medians[i].__getitem__):
	f.write(m + ",")
    f.close()
 
    f = open("../data/" + dset + "/" + labels[i] + "_medians.out", "w")
    for m in rankings[i]:
        f.write(m + "\t" + str(rankings[i][m]) + "\n")
    f.close()

    f = open("../data/" + dset + "/" + labels[i] + "_scores.out", "w")
    for m in scores[i]:
        f.write(m + "\t" + str(scores[i][m]) + "\n")
    f.close()
        
print("Finished")
