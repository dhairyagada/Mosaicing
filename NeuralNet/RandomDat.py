import numpy as np
import pprint
DataSetSize = 10
pp = pprint.PrettyPrinter()

Train_x = np.zeros((DataSetSize,24))
Train_x[:,0:12] = np.random.randint(0,high=100,size=(DataSetSize,12))
Train_x[:,0:12] = 0.1 * Train_x[:,0:12]
Train_y = np.random.randint(0,high=10,size=(DataSetSize,4))
for i in range(DataSetSize):
    Train_x[i,12:15]= Train_y[i,0]*Train_x[i,0:3]
    Train_x[i,15:18]= Train_y[i,1]*Train_x[i,3:6]
    Train_x[i,18:21]= Train_y[i,2]*Train_x[i,6:9]
    Train_x[i,21:24]= Train_y[i,3]*Train_x[i,9:12]
np.save('RandomDataX.npy',Train_x)
np.save('RandomDataY.npy',Train_y)
pX = np.load('RandomDataX.npy')
pY = np.load('RandomDataY.npy')
pp.pprint(pX)
pp.pprint(pY)