import cPickle
import os

os.chdir("C:\Users\Derek\eclipse-workspace\NeuralNetwork\src")
ver = "1"

for i in range(0,18):
    fm = net.test_feature_maps_L1(i)
    for j in range (0,50):
        dir = "../data/" + ver + "/feature_maps/" + str(50*i + j) + "/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        for k in range(0,20):
            np.savetxt(dir + str(k) + ".out", fm[j][k], delimiter="\t")



    
