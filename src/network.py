import sys
import numpy as np
import os
import struct
from array import array as pyarray
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal.pool import pool_2d
from numpy import unique
from random import shuffle
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import copy

# Activation functions for neurons
def linear(z): 
    return z
def ReLU(z): 
    return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
from random import shuffle
from random import seed



#### Constants
GPU = True
if GPU:
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float64'



#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size, c_probs):
        
	sys.setrecursionlimit(1500)	

	self.best_accuracy = 0
        self.current_accuracy = 0
        self.best_auc_roc = 0
        self.best_precision = 0
        self.best_recall = 0
        self.best_f_score = 0
        self.best_predictions = []
        self.best_state = []
        self.best_prob = []
        self.c_probs = c_probs
       
	self.layers = layers
	self.mini_batch_size = mini_batch_size 
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.tensor3("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        self.feature_maps = self.layers[1].conv_out
		
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
			
        #Train the network using mini-batch stochastic gradient descent
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data
		
        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size
		
        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]
				   
        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        train_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })        
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.validate_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
                givens={
                    self.x:
                    validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            }) 
        self.train_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        # Do the actual training
        best_validation_accuracy = 0.0
        best_iteration = 0
        test_accuracy = 0
        for epoch in xrange(epochs):
            batch_order = range(num_training_batches)
            shuffle(batch_order)
            batch_count = 1
            for minibatch_index in batch_order:
                iteration = num_training_batches*epoch+batch_count
                batch_count = batch_count+1
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                    training_accuracy = np.mean(
                        [train_mb_accuracy(j) for j in xrange(num_training_batches)])
                    print("\nEpoch {0}: training accuracy {1:.2%}".format(
                        epoch, training_accuracy))
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    
                    best_iteration = iteration
                        
                    print("Getting Predictions...")
                    probs = []
                    for p in xrange(num_test_batches):
                        probs.append(self.get_prob(p))
                    probs = np.array(probs).reshape(-1,2)
                    probs = [row[1] for row in probs]
                        
                    print("Getting Probabilities...")
                    pred = []
                    for p in xrange(num_test_batches):
                        pred.append(self.test_mb_predictions(p))
                    pred = np.array(pred).reshape(-1,1)

                    true_labels = test_y.eval()
                    auc = roc_auc_score(true_labels, probs)
                    print("Current AUC is " + str(auc))
		    print("Best AUC is " + str(self.best_auc_roc))
		    print(true_labels)
                    print(np.array(pred).reshape(-1))
                    print(probs)
                    if auc >= self.best_auc_roc:
                        precision = precision_score(true_labels, pred, average='weighted')
                        recall = recall_score(true_labels, pred, average='weighted')
                        if auc > self.best_auc_roc or (precision > self.best_precision and recall > self.best_recall): 
                            print("Updating parameters...")
                            self.best_accuracy = validation_accuracy
                            self.best_auc_roc = auc
                            self.best_precision = precision
                            self.best_recall = recall
                            self.best_f_score = f1_score(true_labels, pred, average='weighted')
                            self.best_predictions = pred
                            self.best_prob = probs
                            self.best_state = None
                            self.best_state = copy.copy(self)
                     
                            
                            print('ROC: ' + str(self.best_auc_roc))
                            print('Precision: ' + str(self.best_precision))
                            print('Recall: ' + str(self.best_recall))
                                
        print("Finished training network.")
        print("Best validation AUC of {0:.2%}".format(self.best_auc_roc))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))
        
#### Define layer types
class ConvPoolLayer(object):

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=ReLU):
        
	self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
		
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]
		
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        self.conv_out = theano.tensor.nnet.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            input_shape=self.image_shape)
        pooled_out = pool_2d(input=self.conv_out, ws=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=ReLU, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
		
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]
		
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)
			
    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
		
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]
		
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.velocity = (1-self.p_dropout)*T.dot(self.inpt, self.w)
        self.output = softmax(self.velocity + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)
		
    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean((T.log(self.output_dropout) * net.c_probs)[T.arange(net.y.shape[0]), net.y])
		
    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
