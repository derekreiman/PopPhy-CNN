import cPickle
import os


def save_network(net, dir):
    f = open(dir + "/net.save", 'wb')
    cPickle.dump(net, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def save_network_1D(net, dir):
    f = open(dir + "/net_1D.save", 'wb')
    cPickle.dump(net, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    
