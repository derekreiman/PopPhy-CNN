# Third-party libraries
import numpy as np
import os
import sys
import struct
import argparse

parser = argparse.ArgumentParser(description="PopPhy-CNN Training")
parser.add_argument("-d", "--dataset", default="Cirrhosis",     help="Name of dataset in data folder.")
args = parser.parse_args()

data = args.dataset

fi = {}
fp = open("../data/" + str(data) + "/otu.csv", 'r')
features = fp.readline().split(",")
fp.close()

d = np.loadtxt("../data/" + str(data) + "/count_matrix.csv", delimiter=",")
mean = np.mean(d, axis=0)
sd = np.std(d, axis=0)
snr = mean/sd

for i in range(len(features)):
  fi[features[i]] = snr[i]


fp = open("../data/" + str(data) + "/snr_features.txt", 'w') 
for key, value in sorted(fi.iteritems(), key=lambda(k,v):(v,k), reverse=True):
  fp.write(str(key) + "\t" + str(value) + "\n")
fp.close()

