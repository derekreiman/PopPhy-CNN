import cPickle
import os

def load_data_from_file(dset, norm, method, split):
    
    dir = "../data/" + dset + "/data_sets/" + norm + "/" + method + "/" + split
    
    f = open(dir+"/training.save", 'rb')
    train = cPickle.load(f)
    f.close()
    
    f = open(dir+"/test.save", 'rb')
    test = cPickle.load(f)
    f.close()
    
    f = open(dir+"/validation.save", 'rb')
    validation = cPickle.load(f)
    f.close()
    
    return train, test, validation
    
