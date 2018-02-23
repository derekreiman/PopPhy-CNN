import sys
import pandas as pd
import numpy as np
from load_network import load_network
from load_data import load_data_from_file
from theano.tensor.nnet import conv2d
from graph import Graph, Node
from parse_result import parse_result
dset = sys.argv[1]

acc_list, roc_list, prec_list, recall_list, f1_list, tpr_list, fpr_list, thresh_list = parse_result(dset, "cnn_2D")
max_rocs = np.argsort(roc_list)[::-1]


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

for i in range(0, len(labels)):
	rankings[i] = {}
        scores[i] = {}
	for j in node_names:
	    rankings[i][j] = []
            scores[i][j] = []


for roc in range(0,100):
	roc_index = max_rocs[roc]
        set = str(roc_index / 10)
        cv = str(roc_index % 10)

	net = load_network("../data/" + dset + "/data_sets/raw_noscale/CV_" + set + "/" + cv)
	train, test, validation = load_data_from_file(dset, "raw_noscale", "CV_" + set, cv)

	num_classes = net.layers[-1].w.eval().shape[1]
	num_train = train[1].eval().shape[0]
	num_test = test[1].eval().shape[0]
	num_samp = num_test
	w = net.layers[0].w.eval()
	num_maps = w.shape[1]
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

	num_max = num_nodes
	thresh = 0.5
	thresh_2 = 0.6
	thresh_3 = 0.7

	#Get the top X max indices for each class and each feature map
	max_list = np.zeros((num_classes, num_maps, fm_rows * fm_cols))

	for i in range(0, num_classes):
	    for j in range(0, len(fm[i])):
		for k in range(0, num_maps):
		    maximums = np.argsort(fm[i][j][0][k].flatten())[::-1]
		    for l in range(0, num_max):
		        max_list[i][k][maximums[l]] += 1


	d = {"OTU":[],"Max Score":[], "Cumulative Score":[]}
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
		w = np.rot90(np.rot90(net.layers[0].w.container.data[0][0]))
	    
		#For the top X maximums...
		for k in range(0, len(loc_list)):
		    
		    #Find the row and column location and isolate reference window
		    loc = loc_list[k]    
		    if max_list[i][j][loc] > (len(fm[i]) * 0):
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
		                
		                elif count[m,n] > thresh_2/2:
		                    for r in range((m*w_col) + n + 1, w_row * w_col):
		                        if count[m,n] + count.flatten()[r] > thresh_2 and count.flatten()[r] < thresh and count.flatten()[r] > thresh_2/2:
		                            
		                            val = count[m,n] + count.flatten()[r]
		                            r_row = r / w_col
		                            r_col = r % w_col
		                            pair = ref_window[m,n] + "-" + ref_window[r_row, r_col]

		                            if pair in results[i].index:
		                                if val > results[i].loc[pair, "Max Score"]:
		                                    results[i].loc[pair, "Max Score"] = val
		                                results[i].loc[pair, "Cumulative Score"] += val
		                            else:
		                                results[i].loc[pair, "Max Score"] = val
		                                results[i].loc[pair, "Cumulative Score"] = val
		                        

		                elif count[m,n] > thresh_3/3:
		                    for r in range((m*w_col) + n + 1, w_row * w_col):
		                        for s in range(r + 1, w_row * w_col):
		                            if count[m,n] + count.flatten()[r] + count.flatten()[s] > thresh_3 and count.flatten()[r] < thresh_2/2 and count.flatten()[r] > thresh_3/3 and count.flatten()[s] < thresh_2/2 and count.flatten()[s] > thresh_3/3:

		                                val = count[m,n] + count.flatten()[r] + count.flatten()[s]
		                                r_row = r / w_col
		                                r_col = r % w_col
		                                s_row = s / w_col
		                                s_col = s % w_col
		                                pair = ref_window[m,n] + "-" + ref_window[r_row, r_col] + "-" + ref_windos[s_row, s_col]

		                                if pair in results[i].index:
		                                    if val > results[i].loc[pair, "Max Score"]:
		                                        results[i].loc[pair, "Max Score"] = val
		                                    results[i].loc[pair, "Cumulative Score"] += val
		                                else:
		                                    results[i].loc[pair, "Max Score"] = val
		                                    results[i].loc[pair, "Cumulative Score"] = val

        
	for i in range(0, num_classes):
	    unique = []
	    for feat in results[i].index:
		for sub in feat.split("-"):
		    if sub not in unique:
		        unique.append(sub)
	    for j in range(0, num_classes):
		if i != j:
		    index = results[j].index
		    for feat in index:
		        for sub in feat.split("-"):
		            if sub in unique:
		                unique.remove(sub)

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

  
	features = {}
	graphs = {}
	for i in range(0, num_classes):
	    features[i] = results[i].index
	    graphs[i] = Graph()

	    current_layer = g.layers
	    assigned_layer = False
	    layer_nodes = []

	    while current_layer >= 0:
		node_list = g.get_nodes(current_layer)
	    
		for node in layer_nodes:
		    if node.get_parent != None:
		        parent = g.get_node_by_name(node.get_id()).get_parent()
		        parent_id = parent.get_id()
		        if graphs[i].get_node_by_name(parent_id) == None:
		            new_node = Node(parent_id)
		            new_node.layer = parent.layer
		        else:
		            new_node = graphs[i].get_node_by_name(parent_id)
		        node.set_parent(new_node)
		        new_node.add_child(node)
		        graphs[i].add_node(new_node.layer, new_node)
		        if parent_id in features[i]:
		            index = features[i].get_loc(parent_id)
		            features[i] = features[i].delete(index)
		
	    
		for node in node_list:
		    if node.get_id() in features[i]:
		        new_node = Node(node.get_id())
		        new_node.layer = current_layer
		        if assigned_layer == False:
		            graphs[i].layers = current_layer
		            assigned_layer = True
		        graphs[i].add_node(current_layer, new_node)
		        index = features[i].get_loc(node.get_id())
		        features[i] = features[i].delete(index)
		layer_nodes = graphs[i].get_nodes(current_layer)
		current_layer -= 1
	    
	    


	#for i in range(0, num_classes):    
	    #graphs[i].write_table("../data/" + dset + "/" + labels[i] + "-" + set +"-" + cv + "_graph.out")
	    #results[i].to_csv("../data/" + dset + "/" + labels[i] + "-" + set +"-" + cv + "_results.out")

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
