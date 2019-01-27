#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pprint

pp = pprint.PrettyPrinter()
## Training Dataset 
DataSetSize = 50
x = np.zeros((DataSetSize,48))
x[:,0:24] = np.random.rand(10,24)
y = np.random.randint(0,high=10,size=(48,3))
for i in range(DataSetSize):
    x[i,24:32]= y[i,0]*x[i,0:8]
    x[i,32:40]= y[i,1]*x[i,8:16]
    x[i,40:48]= y[i,2]*x[i,16:24]