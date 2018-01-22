import sys
import pandas as pd
import numpy as np
from load_network import load_network
from load_data import load_data_from_file
from theano.tensor.nnet import conv2d
from graph import Graph, Node
from parse_result import parse_result
import plotly
import plotly.plotly as py
import plotly.graph_objs as go


def min_max_norm(x):
	return (x/x.max())



plotly.tools.set_credentials_file(username='dreima2', api_key='Lxv7WmZD3IuB90B7QcAB')

dset = sys.argv[1]

acc_list, roc_list, prec_list, recall_list, f1_list, tpr_list, fpr_list, thresh_list = parse_result(dset, "cnn")
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

fm_mean = {}
for i in range(len(labels)):
    fm_mean[i] = []

for set in range(10):
    for cv in range(10):
	
	net = load_network("../data/" + dset + "/data_sets/CV_" + str(set) + "/" + str(cv))
	train, test, validation = load_data_from_file(dset, "CV_" + str(set), str(cv))
	num_classes = net.layers[-1].w.eval().shape[1]
	num_train = train[1].eval().shape[0]
	num_test = test[1].eval().shape[0]
	num_samp = num_test
	w = net.layers[0].w.eval()
	num_maps = w.shape[1]
	w_row = w.shape[2]
	w_col = w.shape[3]

	data_set = test
	data_shape = data_set[0].eval().shape
	
	fm = {}

 	for i in range(2):
	    fm[i] = []

	for i in range(0, num_samp):
    		f = conv2d(data_set[0][i].reshape([1,1,data_shape[1], data_shape[2]]), w)
		y = data_set[1].eval()[i]
    		fm[int(y)].append(f.eval())

        for i in range(2):
		m = np.mean(np.sum(np.array(fm[i]),axis=2).clip(min=0),axis=0)
		fm_mean[i].append(m)

fm_median = {}
fm_median[0] = min_max_norm(np.mean(fm_mean[0], axis=0))
fm_median[1] = min_max_norm(np.mean(fm_mean[1], axis=0))

fm_diff = np.subtract(fm_median[1], fm_median[0])
fm_diff_mean = min_max_norm(np.mean(fm_diff, axis=0))

print(fm_diff_mean)

trace = go.Heatmap(z=fm_diff_mean)
data=[trace]
py.iplot(data, filename=dset + "_diff")

#for i in range(2):
#	trace = go.Heatmap(z=fm_median[i][0])
#	data = [trace]
#	py.iplot(data, filename=dset + "_" + labels[i])

