import sys
import network
import argparse
from network import Network
from network import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network import ReLU
from math import ceil, floor, sqrt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from fractions import gcd
from popphy_io import load_data_from_file, save_network
import numpy as np
from theano.tensor import tanh


parser = argparse.ArgumentParser(description="Prepare Data")
parser.add_argument("-e", "--epochs", default=400, 	type=int, help="Number of epochs.")
parser.add_argument("-b", "--batch_size", default=1, type=int, 	help="Batch Size")
parser.add_argument("-n", "--splits", default=10, type=int, help="Number of cross validated splits.")
parser.add_argument("-s", "--sets", default=10, type=int, help="Number of datasets")
parser.add_argument("-d", "--dataset", default="Cirrhosis", 	help="Name of dataset in data folder.")

args = parser.parse_args()

dset = args.dataset
num_epochs = args.epochs
mini_batch_size = args.batch_size

net_best_accuracy = []
net_best_roc = []
net_best_precision = []
net_best_recall = []
net_best_f_score = []
net_best_predictions = []
net_best_probs = []

for set in range(0, args.sets):
    for cv in range(0,args.splits):
	print("\nRun " + str(cv) + " of set " + str(set) + ".\n")
        train, test, validation = load_data_from_file(dset, "CV_" + str(set), str(cv))
         
        train_lab = train[1].eval()
        c_prob = [None] * len(np.unique(train_lab))
        for l in np.unique(train_lab):
            c_prob[l] = float( float(len(train_lab))/ (np.sum(train_lab == l)))
            
        dir = "../data/" + dset + "/data_sets/"  + "CV_" + str(set) + "/" + str(cv)
        rows = train[0].container.data.shape[1]
        cols = train[0].container.data.shape[2]

	#Dataset specific models		
        if dset == "T2D":
               
            net = Network([
                ConvPoolLayer(activation_fn=ReLU, image_shape=(mini_batch_size, 1, rows, cols), 
                            filter_shape=(64, 1, 5, 9), 
                            poolsize=(2, 2)),  
                ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 11, 53), 
                            filter_shape=(64, 64, 4, 10), 
                            poolsize=(2, 2)),
                ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 4, 22), 
                            filter_shape=(64, 64, 3, 9),
                            poolsize=(2,2)),  
                FullyConnectedLayer(activation_fn=ReLU, n_in=64*1*7, n_out=1024, p_dropout=0.5),
                FullyConnectedLayer( n_in=1024, n_out=1024, activation_fn=ReLU, p_dropout=0.5),
                SoftmaxLayer(n_in=1024, n_out=2, p_dropout=0.5)], mini_batch_size, c_prob)
            
        if dset == "Obesity":
                
            net = Network([
                ConvPoolLayer(activation_fn=ReLU, image_shape=(mini_batch_size, 1, rows, cols), 
                            filter_shape=(64, 1, 5, 10), 
                            poolsize=(2, 2)),  
                ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 9, 47), 
                            filter_shape=(64, 64, 4, 10), 
                            poolsize=(2, 2)),
                ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 3, 19), 
                            filter_shape=(64, 64, 3, 10),
                            poolsize=(1,2)),  
                FullyConnectedLayer(activation_fn=ReLU, n_in=64*1*5, n_out=1024, p_dropout=0.5),
                FullyConnectedLayer( n_in=1024, n_out=1024, activation_fn=ReLU, p_dropout=0.5),
                SoftmaxLayer(n_in=1024, n_out=2, p_dropout=0.5)], mini_batch_size, c_prob)
                 
        if dset == "Cirrhosis":
                
            net = Network([
                ConvPoolLayer(activation_fn=ReLU, image_shape=(mini_batch_size, 1, rows, cols), 
                            filter_shape=(64, 1, 5, 10), 
                            poolsize=(2, 2)),  
                ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 11, 47), 
                            filter_shape=(64, 64, 4, 10), 
                            poolsize=(2, 2)),
                ConvPoolLayer(activation_fn=ReLU,image_shape=(mini_batch_size, 64, 4, 19), 
                            filter_shape=(64, 64, 3, 10),
                            poolsize=(2,2)),  
                FullyConnectedLayer(activation_fn=ReLU, n_in=64*1*5, n_out=1024, p_dropout=0.5),
                FullyConnectedLayer( n_in=1024, n_out=1024, activation_fn=ReLU, p_dropout=0.5),
                SoftmaxLayer(n_in=1024, n_out=2, p_dropout=0.5)], mini_batch_size, c_prob)
                 
        net.SGD(train, num_epochs, mini_batch_size, 0.001, 
            validation, test, lmbda=0.1)
            
        net_best_accuracy.append(net.best_accuracy)
        net_best_roc.append(net.best_auc_roc)
        net_best_precision.append(net.best_precision)
        net_best_recall.append(net.best_recall)
        net_best_f_score.append(net.best_f_score)
        net_best_predictions.append(net.best_predictions)
        net_best_probs.append(net.best_prob)
        save_network(net.best_state, dir)
        
#Record Results

dir = "../data/" + dset + "/data_sets/"    
f = open(dir + "/" + str(num_epochs) + "_results.txt", 'w')
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
       
for i in range(0,args.sets * args.splits):
    f.write("\nPredictions for " + str(i) + "\n")
    f.write("\n" + str(net_best_predictions[i]) + "\n")
    f.write("\n" + str(net_best_probs[i]) + "\n")
f.close()
