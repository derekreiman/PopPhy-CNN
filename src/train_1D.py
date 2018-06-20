import sys
import network
import argparse
from network import Network
from network import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network import ReLU
from math import ceil, floor, sqrt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from fractions import gcd
from popphy_io import load_data_from_file, save_network_1D
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from theano.tensor import tanh

sys.setrecursionlimit(10000)

def shared2(datax, datay):
	return datax, T.cast(datay, "int32")


parser = argparse.ArgumentParser(description="PopPhy-CNN Training")
parser.add_argument("-e", "--epochs", default=400,      type=int, help="Number of epochs.")
parser.add_argument("-b", "--batch_size", default=1, type=int,  help="Batch Size")
parser.add_argument("-n", "--splits", default=10, type=int, help="Number of cross validated splits.")
parser.add_argument("-s", "--sets", default=10, type=int, help="Number of datasets")
parser.add_argument("-d", "--dataset", default="Cirrhosis",     help="Name of dataset in data folder.")
parser.add_argument("-m", "--method", default="CV", help="CV or holdout.")
parser.add_argument("-p", "--patience", default=-1, help="Epoch count for early stopping.")
parser.add_argument("-r", "--dropout", default=0.5, help="Dropout rate.")
parser.add_argument("-h1", "--hidden_1", default=1024, help="Nodes in first fully connected layer")
parser.add_argument("-h2", "--hidden_2", default=1024, help="Nodes in second fully connected layer")
args = parser.parse_args()

dset = args.dataset
num_epochs = args.epochs
mini_batch_size = args.batch_size
dim1 = args.hidden_1
dim2 = args.hidden_2
dropout = args.dropout
patience = args.patience
num_sets = args.sets

net_best_accuracy = []
net_best_roc = []
net_best_precision = []
net_best_recall = []
net_best_f_score = []
net_best_predictions = []
net_best_probs = []

if args.method == "CV":
    prefix = "CV_"
    num_splits = args.splits

if args.method == "holdout" or args.method == "HO":
    prefix = "HO_"
    num_splits=1

for set in range(0, num_sets):
	for cv in range(0, num_splits):
		print("\nRun " + str(cv) + " of set " + str(set) + ".\n")
		seed = np.random.randint(0, 100) 
		dir = "../data/" + dset + "/data_sets/" + prefix + str(set) + "/" + str(cv)
		x = pd.read_csv(dir+"/benchmark_train_data.csv", header=None, dtype=np.float64)
		y = np.asarray(np.loadtxt(dir+'/benchmark_train_labels.csv', delimiter=','))
		tx = pd.read_csv(dir+"/benchmark_test_data.csv", header=None, dtype=np.float64)
		ty = np.asarray(np.loadtxt(dir+'/benchmark_test_labels.csv', delimiter=','))

		x = np.asarray(x)
		tx = np.asarray(tx)		
		x = x.reshape(x.shape[0], 1, x.shape[1])
		tx = tx.reshape(tx.shape[0], 1, tx.shape[1])

		x_train = theano.shared(x)
		y_train = theano.shared(y)
		x_test = theano.shared(tx)
		y_test = theano.shared(ty)
		train = shared2(x_train,y_train)
		test = shared2(x_test, y_test)
		validation = shared2(x_test, y_test)

		train_lab = train[1].eval()
		c_prob = [None] * len(np.unique(train_lab))
		for l in np.unique(train_lab):
			c_prob[l] = float( float(len(train_lab))/ (np.sum(train_lab == l)))
			
		rows = 1
		cols = train[0].container.data.shape[2]
		
		if dset == "T2D":
			   
			net = Network([
				ConvPoolLayer(activation_fn=ReLU, image_shape=(mini_batch_size, 1, rows, cols), 
							filter_shape=(64, 1, 1, 10), 
							poolsize=(1, 2)),  
				ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 1, 102), 
							filter_shape=(64, 64, 1, 10), 
							poolsize=(1, 2)),
				ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 1, 46), 
							filter_shape=(64, 64, 1, 10),
							poolsize=(1,2)),  
				FullyConnectedLayer(activation_fn=ReLU, n_in=64*1*18, n_out=dim1, p_dropout=dropout),
				FullyConnectedLayer( n_in=dim1, n_out=dim2, activation_fn=ReLU, p_dropout=dropout),
				SoftmaxLayer(n_in=dim2, n_out=2, p_dropout=dropout)], mini_batch_size, c_prob)
			
		if dset == "Obesity":
				
			net = Network([
				ConvPoolLayer(activation_fn=ReLU, image_shape=(mini_batch_size, 1, rows, cols), 
							filter_shape=(64, 1, 1, 10), 
							poolsize=(1, 2)),  
				ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 1, 86), 
							filter_shape=(64, 64, 1, 10), 
							poolsize=(1, 2)),
				ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 1, 38), 
							filter_shape=(64, 64, 1, 10),
							poolsize=(1,2)),  
				FullyConnectedLayer(activation_fn=ReLU, n_in=64*1*14, n_out=dim1, p_dropout=dropout),
				FullyConnectedLayer( n_in=dim1, n_out=dim2, activation_fn=ReLU, p_dropout=dropout),
				SoftmaxLayer(n_in=dim2, n_out=2, p_dropout=dropout)], mini_batch_size, c_prob)
				 
		if dset == "Cirrhosis":
				
			net = Network([
				ConvPoolLayer(activation_fn=ReLU, image_shape=(mini_batch_size, 1, rows, cols), 
							filter_shape=(64, 1, 1, 10), 
							poolsize=(1, 2)),  
				ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 1, 87), 
							filter_shape=(64, 64, 1, 10), 
							poolsize=(1, 2)),
				ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 1, 39), 
							filter_shape=(64, 64, 1, 10),
							poolsize=(1,2)),  
				FullyConnectedLayer(activation_fn=ReLU, n_in=64*1*15, n_out=dim1, p_dropout=dropout),
				FullyConnectedLayer( n_in=dim1, n_out=dim2, activation_fn=ReLU, p_dropout=dropout),
				SoftmaxLayer(n_in=dim2, n_out=2, p_dropout=dropout)], mini_batch_size, c_prob)
				
				 
		net.SGD(train, num_epochs, mini_batch_size, 0.001, 
			validation, test, lmbda=0.1, patience=patience)
			
		net_best_accuracy.append(net.best_accuracy)
		net_best_roc.append(net.best_auc_roc)
		net_best_precision.append(net.best_precision)
		net_best_recall.append(net.best_recall)
		net_best_f_score.append(net.best_f_score)
		net_best_predictions.append(net.best_predictions)
		net_best_probs.append(net.best_prob)
		save_network_1D(net.best_state, dir)
	  
dir = "../data/" + dset + "/data_sets/"
f = open(dir + "/" + prefix + "results_1D_CNN.txt", 'w')
f.write("Mean Accuracy: " + str(np.mean(net_best_accuracy)) + " (" + str(np.std(net_best_accuracy)) + ")\n")
f.write(str(net_best_accuracy) + "\n")
f.write("\nMean ROC: " + str(np.mean(net_best_roc)) + " (" + str(np.std(net_best_roc)) + ")\n")
f.write(str(net_best_roc) + "\n")
f.write("\nMean Precision: " + str(np.mean(net_best_precision)) + " (" + str(np.std(net_best_precision)) + ")\n")
f.write(str(net_best_precision) + "\n")
f.write("\nMean Recall: " + str(np.mean(net_best_recall)) + " (" + str(np.std(net_best_recall)) + ")\n")
f.write(str(net_best_recall) + "\n")
f.write("\nMean F-score: " + str(np.mean(net_best_f_score)) + " (" + str(np.std(net_best_f_score)) + ")\n")
f.write(str(net_best_f_score) + "\n")
	   
for i in range(0,num_splits * num_sets):
	f.write("\nPredictions for " + str(i) + "\n")
	f.write("\n" + str(list(np.array(net_best_predictions[i]).reshape(-1))) + "\n")
	f.write("\n" + str(list(np.array(net_best_probs[i]).reshape(-1))) + "\n")
f.close()
