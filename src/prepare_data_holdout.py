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
    new_in = np.add(x, np.random.normal(0,0.0000005, num_features)).clip(min=0)
    g.populate_graph(f, new_in)
    return new_in, np.array(g.get_map())


data_sets = ["T2D", "MovingPictures"]
num_cores = multiprocessing.cpu_count()

for data in data_sets:
    
    os.chdir("/home/derek/workspace/NeuralNetwork/data/" + data)
    print("Processing " + data +"...")
    
    relative = False
    if data == 'T2D':
        relative = True
        print("Using relative abundance...")
    
        
    #Data Variables    
    my_x = []
    my_y = []

        
    #Get count matrices
    print ("Opening data files...") 
    my_x = np.loadtxt('count_matrix.csv', dtype=np.float32, delimiter=',')
        
    #Get sample labels    
    my_y = np.genfromtxt('labels.txt', dtype=np.str_)
    
    #Get the list of OTUs            
    features = np.genfromtxt('otu.csv', dtype=np.str_, delimiter=',')
    
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
    
    #Set size to scale data up to
    train_size = 500
    test_size = 200
    validation_size = 200
    
    #Build phylogenetic tree graph
    g = Graph()
    g.build_graph("newick.txt")
    print("Graph constructed...")
    
    #Create 10 CV sets
    for set in range(0,10):
        print("Generating set " + str(set) + "...")
        
        my_train_data = []
        my_train_labels = []
        my_validation_data = []
        my_validation_labels = []
        my_test_data = []
        my_test_labels = []
        benchmark_train_data = []
        benchmark_train_labels = []
        benchmark_test_data = []
        benchmark_test_labels = []
        
        set_seed = np.random.randint(0, 2000)
        np.random.seed(set_seed)
        np.random.shuffle(my_x)
        np.random.seed(set_seed)
        np.random.shuffle(my_y)
    
        #For each class
        for c in range(0,num_classes):
            print("Class " + str(classes[c]) + "...")
            
            #Get the data indeces where that class is found
            ind = np.where(my_y == classes[c])[0]
            
            #Split indeces into training, testing, and validation
            train_ind = ind[0:int(round(len(ind) * 0.95))]
            test_ind = ind[int(round(len(ind) *0.95) + 1):int(round(len(ind) * 0.99))]
            validation_ind = ind[int(round(len(ind) * 0.95) + 1):len(ind)]
        
            #For each set (train, test, validation), add a little noise before populating the graph and add it to the dataset
            index = np.random.choice(train_ind, train_size)
            results = Parallel(n_jobs=num_cores)(delayed(generate_maps)(x,g,features) for x in my_x[index])
            my_train_data.append(np.array(np.take(results,1,1).tolist()))
            benchmark_train_data.append(np.array(np.take(results,0,1).tolist()))
            my_train_labels = np.concatenate((my_train_labels, np.repeat(int(c), train_size)), axis=0)
            benchmark_train_labels = np.concatenate((benchmark_train_labels, np.repeat(int(c), train_size)), axis=0)
                                         
            index = np.random.choice(test_ind, test_size)
            results = Parallel(n_jobs=num_cores)(delayed(generate_maps)(x,g,features) for x in my_x[index])
            my_test_data.append(np.array(np.take(results,1,1).tolist()))
            benchmark_test_data.append(np.array(np.take(results,0,1).tolist()))
            my_test_labels = np.concatenate((my_test_labels, np.repeat(int(c), test_size)), axis=0)
            benchmark_test_labels = np.concatenate((benchmark_test_labels, np.repeat(int(c), test_size)), axis=0)
                                         
            index = np.random.choice(validation_ind, validation_size)
            results = Parallel(n_jobs=num_cores)(delayed(generate_maps)(x,g,features) for x in my_x[index])
            my_validation_data.append(np.array(np.take(results,1,1).tolist()))
            benchmark_test_data.append(np.array(np.take(results,0,1).tolist()))
            my_validation_labels = np.concatenate((my_validation_labels, np.repeat(int(c), validation_size)), axis=0)
            benchmark_test_labels = np.concatenate((benchmark_test_labels, np.repeat(int(c), validation_size)), axis=0)
        

        
        my_train_data = np.array(my_train_data, dtype=np.float64)
        my_train_labels = np.array(my_train_labels, dtype=np.int32)
        my_test_data = np.array(my_test_data, dtype=np.float64)
        my_test_labels = np.array(my_test_labels, dtype=np.int32)
        my_validation_data = np.array(my_validation_data, dtype=np.float64)
        my_validation_labels = np.array(my_validation_labels, dtype=np.int32)
        benchmark_train_data = np.array(benchmark_train_data, dtype=np.float64)
        benchmark_train_labels = np.array(benchmark_train_labels, dtype=np.int32)
        benchmark_test_data = np.array(benchmark_test_data, dtype=np.float64)
        benchmark_test_labels = np.array(benchmark_test_labels, dtype=np.int32)
        
        map_rows = my_train_data.shape[2]
        map_cols = my_train_data.shape[3]
        
        my_train_data = my_train_data.reshape(num_classes * train_size, map_rows, map_cols)
        my_train_labels = my_train_labels.reshape(num_classes*train_size)
        my_test_data = my_test_data.reshape(num_classes * test_size, map_rows, map_cols)
        my_test_labels = my_test_labels.reshape(num_classes * test_size)
        my_validation_data = my_validation_data.reshape(num_classes * validation_size, map_rows, map_cols)
        my_validation_labels = my_validation_labels.reshape(num_classes*validation_size)
        benchmark_train_data = benchmark_train_data.reshape(num_classes * train_size, num_features)
        benchmark_train_labels = benchmark_train_labels.reshape(num_classes * train_size)
        benchmark_test_data = benchmark_test_data.reshape(num_classes * (test_size + validation_size), num_features)
        benchmark_test_labels = benchmark_test_labels.reshape(num_classes * (test_size + validation_size))
        
        #Random seeds before shuffling
        train_seed = np.random.randint(0,100)
        test_seed = np.random.randint(0,100)
        validation_seed = np.random.randint(0,100)
        
        #Shuffle training set
        np.random.seed(train_seed)
        np.random.shuffle(my_train_data)
        np.random.seed(train_seed)
        np.random.shuffle(my_train_labels)
        np.random.seed(train_seed)
        np.random.shuffle(benchmark_train_data)
        np.random.seed(train_seed)
        np.random.shuffle(benchmark_train_labels)
        
        #Shuffle test set
        np.random.seed(test_seed)
        np.random.shuffle(my_test_data)
        np.random.seed(test_seed)
        np.random.shuffle(my_test_labels)
        np.random.seed(test_seed)
        np.random.shuffle(benchmark_test_data)
        np.random.seed(test_seed)
        np.random.shuffle(benchmark_test_labels)
        
        #Shuffle validation set
        np.random.seed(validation_seed)
        np.random.shuffle(my_validation_data)
        np.random.seed(validation_seed)
        np.random.shuffle(my_validation_labels)
        
        clf.fit(benchmark_train_data, benchmark_train_labels)
        print(clf.score(benchmark_test_data, benchmark_test_labels))

        
        #Create Theano shared objects
        training_set_x = theano.shared(my_train_data)
        training_set_y = theano.shared(np.squeeze(my_train_labels))
        validation_set_x = theano.shared(my_validation_data)
        validation_set_y = theano.shared(np.squeeze(my_validation_labels))
        test_set_x = theano.shared(my_test_data)
        test_set_y = theano.shared(np.squeeze(my_test_labels))
        print("Finished Setup...") 
        
        #Combine data and labels into a single object
        train = shared2(training_set_x,training_set_y)
        validation = shared2(validation_set_x, validation_set_y)
        test = shared2(test_set_x, test_set_y)
        
        dir = "data_sets/" + str(set)
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
        cPickle.dump(validation, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        
        np.savetxt(dir + "/benchmark_train_data.csv", benchmark_train_data, delimiter=',')
        np.savetxt(dir + "/benchmark_train_labels.csv", benchmark_train_labels, delimiter=',')
        np.savetxt(dir + "/benchmark_test_data.csv", benchmark_test_data, delimiter=',')
        np.savetxt(dir + "/benchmark_test_labels.csv", benchmark_test_labels, delimiter=',')
        
        print("Finished writing files...")
        
        
