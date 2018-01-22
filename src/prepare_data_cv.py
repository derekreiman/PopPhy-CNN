import numpy as np
import os
import struct
from array import array as pyarray
from numpy import unique
import cPickle
from graph import Graph
from random import shuffle
from random import seed
from joblib import Parallel, delayed
import multiprocessing
import time
from sklearn.cross_validation import StratifiedKFold
import pandas as pd

import theano
import theano.tensor as T

#Create Theano shared variables
def shared(data):
    shared_x = theano.shared(
        np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(
        np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
    return shared_x, T.cast(shared_y, "int32")
        
def shared2(datax, datay):
    return datax, T.cast(datay, "int32")

def generate_maps(x, g, f):
    id = multiprocessing.Process()._identity
    np.random.seed((int(time.time()) + id[1]^2)/id[0])
    new_in = np.add(x, np.random.normal(0,0.00005, num_features)).clip(min=0).clip(max=1)
    g.populate_graph(f, x)
    return x, np.array(g.get_map())


data_sets = ["T2D"]
num_cores = multiprocessing.cpu_count()

for data in data_sets:
    for set in range(0,10):
        scale = 1
        dat_dir = "../data/" + data
        print("Processing " + data +"...")
        
        relative = False
        if data == 'T2D' or data=='Obesity' or data=="Cirrhosis" or data=="Colorectal":
            relative = True
            print("Using relative abundance...")
        
            
        #Data Variables    
        my_x = []
        my_y = []
    
            
        #Get count matrices
        print ("Opening data files...") 
        my_x = np.loadtxt(dat_dir + '/count_matrix.csv', dtype=np.float32, delimiter=',')
            
        #Get sample labels    
        my_y = np.genfromtxt(dat_dir + '/labels.txt', dtype=np.str_, delimiter=',')
        
        #Get the list of OTUs            
        features = np.genfromtxt(dat_dir+ '/otu.csv', dtype=np.str_, delimiter=',')
        
        print("Finished reading data...")
        
        #Filter out samples with low total read counts
        ######## TODO: Create threshold calculation dynamically ###########
        if  not relative: 
            my_y = my_y[(np.sum(my_x, 1) > 1000)]
            my_x = my_x[(np.sum(my_x, 1) > 1000)]    
            
        num_samples = my_x.shape[0]
        num_features = len(my_x[0])
        
        #Get the set of classes    
        classes = list(unique(my_y))
        num_classes = len(classes)
        print("There are " + str(num_classes) + " classes")  
        my_ref = pd.factorize(my_y)[1]
        f = open(dat_dir + "/label_reference.txt", 'w')
        f.write(str(my_ref))
        f.close()
        
        #Build phylogenetic tree graph
        g = Graph()
        g.build_graph(dat_dir + "/newick.txt")
        print("Graph constructed...")
        #Create 10 CV sets
        print("Generating set " + str(set) + "...")
        
        my_data = pd.DataFrame(my_x)
        #my_data = (my_data - my_data.min())/(my_data.max() - my_data.min())
        my_data = np.array(my_data)
	my_lab = pd.factorize(my_y)[0]
        my_maps = []
        my_benchmark = []
        
        for s in range(0,scale):            
            #For each set (train, test, validation), add a little noise before populating the graph and add it to the dataset
            results = Parallel(n_jobs=num_cores)(delayed(generate_maps)(x,g,features) for x in my_data)
            my_maps.append(np.array(np.take(results,1,1).tolist()))
            my_benchmark.append(np.array(np.take(results,0,1).tolist()))
                                         
        my_maps = np.array(my_maps)
        my_benchmark = np.array(my_benchmark)                
        print(my_maps.shape) 
	map_rows = my_maps.shape[2]
	map_cols = my_maps.shape[3]

        
      
        print("Finished Setup...") 
        k_fold = StratifiedKFold(my_lab, n_folds=10, shuffle=True)
        
        count = 0
        #for train_index, test_index in kf.split(my_maps[0], my_lab):
        for train_index, test_index in k_fold:
            x_train = []
            x_test = []
            y_train=[]
            y_test=[]
            benchmark_train=[]
            benchmark_test=[]
            
            print("Creating split " + str(count))
            
            for s in range(0,scale):
		print(s)
		print(train_index)
                x_train.append(my_maps[s][train_index])
                x_test.append(my_maps[s][test_index])
                y_train.append(my_lab[train_index])
                y_test.append(my_lab[test_index])
                benchmark_train.append(my_benchmark[s][train_index])
                benchmark_test.append(my_benchmark[s][test_index])
            
 
            x_train = np.array(x_train).reshape(-1, map_rows, map_cols)
            x_test = np.array(x_test).reshape(-1, map_rows, map_cols)
            y_train = np.squeeze(np.array(y_train).reshape(1,-1), 0)
            y_test = np.squeeze(np.array(y_test).reshape(1,-1), 0)
            benchmark_train = np.array(benchmark_train).reshape(-1, num_features)
            benchmark_test = np.array(benchmark_test).reshape(-1, num_features)
           

            seed = np.random.randint(1000)
	    np.random.seed(seed)
	    np.random.shuffle(x_train)
	    np.random.seed(2*seed)
	    np.random.shuffle(x_test)
            np.random.seed(seed)
            np.random.shuffle(y_train)
            np.random.seed(2*seed)
            np.random.shuffle(y_test)
            np.random.seed(seed)
            np.random.shuffle(benchmark_train)
            np.random.seed(2*seed)
            np.random.shuffle(benchmark_test)



 
            #Combine data and labels into a single object
            x_train = theano.shared(x_train)
            y_train2 = theano.shared(y_train)
            x_test = theano.shared(x_test)
            y_test2 = theano.shared(y_test)
            train = shared2(x_train,y_train2)
            test = shared2(x_test, y_test2)
            
            dir = dat_dir + "/data_sets/raw_noscale/CV_" + str(set)
            if not os.path.exists(dir):
                os.makedirs(dir)
                
            dir = dat_dir + "/data_sets/raw_noscale/CV_" + str(set) + "/" + str(count)
            if not os.path.exists(dir):
                os.makedirs(dir)
            
            #Save the data sets in Pickle format
            f = open(dir + "/training.save", 'wb')
            cPickle.dump(train, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            
            f = open(dir + "/test.save", 'wb')
            cPickle.dump(test, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            
            f = open(dir + "/validation.save", 'wb')
            cPickle.dump(test, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
            
        
            np.savetxt(dir + "/benchmark_train_data.csv", benchmark_train, delimiter=',')
            np.savetxt(dir + "/benchmark_train_labels.csv", y_train, delimiter=',')
            np.savetxt(dir + "/benchmark_test_data.csv", benchmark_test, delimiter=',')
            np.savetxt(dir + "/benchmark_test_labels.csv", y_test, delimiter=',')
            count = count + 1
        
        print("Finished writing files...")
        
        
