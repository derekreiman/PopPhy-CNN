import os
import numpy as np

ver = "1"
os.chdir("C:\Users\Derek\eclipse-workspace\NeuralNetwork\src")
num_maps = 20
batches = 18
batch_size = 50
fm = []

for i in range(0,18):
    for j in range (0,50):
        dir = "../data/" + ver + "/feature_maps-64/" + str(50*i + j) + "/"
        maps = []
        for k in range(0,num_maps):
            m = np.loadtxt(dir + str(k) + ".out", delimiter="\t")
            maps.append(m)
        fm.append(maps)

fm = np.array(fm)