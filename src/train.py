import sys
import network
from network import Network
from network import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network import ReLU
from math import ceil, floor, sqrt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from fractions import gcd
from load_data import load_data_from_file
from save_network import save_network
import numpy as np
from theano.tensor import tanh

k1_width = 7
k1_height = 7
k2_width = 7
k2_height = 7
k3_width = 3
k3_height = 3

num_epochs = int(sys.argv[4])

dset = sys.argv[1]
norm = sys.argv[2]
mini_batch_size = int(sys.argv[3])

net_best_accuracy = []
net_best_roc = []
net_best_precision = []
net_best_recall = []
net_best_f_score = []
net_best_predictions = []
net_best_probs = []

for set in range(0, 10):
    for cv in range(0,10):
	print("\nRun " + str(cv) + " of set " + str(set) + ".\n")
        train, test, validation = load_data_from_file(dset, norm, "CV_" + str(set), str(cv))
         
        train_lab = train[1].eval()
        c_prob = [None] * len(np.unique(train_lab))
        for l in np.unique(train_lab):
            c_prob[l] = float( float(len(train_lab))/ (np.sum(train_lab == l)))
            
        dir = "../data/" + dset + "/data_sets/" + norm + "/CV_" + str(set) + "/" + str(cv)
        rows = train[0].container.data.shape[1]
        cols = train[0].container.data.shape[2]
        L1_height = int(floor((rows-k1_height+1)/2.0))
        L1_width = int(floor((cols-k1_width+1)/2.0))
        L2_height = int(floor((L1_height-k2_height+1)/1))
        L2_width = int(floor((L1_width-k2_width+1)/2.0))
        L3_height = int(floor((L2_height-k3_height+1)))
        L3_width = int(floor((L2_width-k3_width+1)/2.0))

	print(rows)
	print(cols)        

        if dset == "T2D":
            #mini_batch_size = 4
               
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
            #mini_batch_size = 4
                
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
            #mini_batch_size = 4
                
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
        
dir = "../data/" + dset + "/data_sets/" + norm    
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
       
for i in range(0,100):
    f.write("\nPredictions for " + str(i) + "\n")
    f.write("\n" + str(np.array(net_best_predictions[i]).reshape(1,-1)) + "\n")
    f.write("\n" + str(np.array(net_best_probs[i]).reshape(1, -1)) + "\n")
f.close()
