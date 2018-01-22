import cPickle
import numpy as np
import os

os.chdir("C:\Users\Derek\eclipse-workspace\NeuralNetwork\src")
ver = "1"

for i in range(0,18):
    sal = net.get_saliency(i)
    for j in range (0,50):
        np.savetxt("../data/" + ver + "/saliency_maps/gut/" + str(50*i + j) + ".out", sal[3*j][j][0], delimiter="\t")
        np.savetxt("../data/" + ver + "/saliency_maps/skin/" + str(50*i + j) + ".out", sal[3*j+1][j][0], delimiter="\t")
        np.savetxt("../data/" + ver + "/saliency_maps/oral/" + str(50*i + j) + ".out", sal[3*j+2][j][0], delimiter="\t")


    