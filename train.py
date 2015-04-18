#!/home/ashidham/anaconda/bin/python2.7

import os, h5py, caffe, subprocess
import sklearn.cross_validation
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr

X = np.loadtxt(open("X_sample.csv","rb"),delimiter=",")
y = np.loadtxt(open("Y_sample.csv","rb"),delimiter=",")
X, Xt, y, yt = sklearn.cross_validation.train_test_split(X, y)
with h5py.File("train.h5","w") as f:
	f['data'] = np.reshape(X,(292,1,320,1))
	f['label'] = y
with h5py.File("test.h5","w") as f:
	f['data'] = np.reshape(Xt,(98,1,320,1))
	f['label'] = yt
os.system("$CAFFE_ATAC train -solver h5.prototxt")
