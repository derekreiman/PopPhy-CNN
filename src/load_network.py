import cPickle
import os

def load_network(dir):
    f = open(dir+"/net.save", 'rb')
    dat = cPickle.load(f)
    f.close()
    return dat

def load_network_1D(dir):
    f = open(dir + "/net_1D.save", "rb")
    dat = cPickle.load(f)
    f.close()
    return dat    
