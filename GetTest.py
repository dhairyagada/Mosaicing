import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from numpy.linalg import inv
pathA = './ImageProc/InputImages/test1.jpg'
pathB = './ImageProc/InputImages/test2.jpg'
imgA = cv2.imread(pathA)
imgB = cv2.imread(pathB)

from mynetconfig import *
from glob import glob
import os
from pylab import *

pathA = './ImageProc/InputImages/test1.jpg'
pathB = './ImageProc/InputImages/test2.jpg'
imgA = cv2.imread(pathA)
imgB = cv2.imread(pathB)

imgA = cv2.resize(imgA,(w,h))
imgB = cv2.resize(imgB,(w,h))

A = cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)
B = cv2.cvtColor(imgB,cv2.COLOR_RGB2GRAY)

A = imgA[0:h,w-180:w]
B = imgB[0:h,0:180]

xright = randint(rho,x_l-rho-patchsize)
yright = randint(rho,h-rho-patchsize-newpointdel)
#cv2.rectangle(A,(rho,rho),(x_l-rho-patchsize,h-rho-patchsize),(0,255,255),2)
#cv2.rectangle(B,(rho,rho),(x_l-rho-patchsize,h-rho-patchsize),(0,255,255),2)

upleft = (xright,yright)
botleft = (xright,yright+patchsize)
botright = (xright+patchsize,yright+patchsize)
upright = (xright+patchsize,yright)

points = [upleft,botleft,botright,upright]

points = np.array(points)
points = points.reshape((1,4,2))
#A = cv2.polylines(A,points,1,(0,0,255),2)
#B = cv2.polylines(B,points,1,(0,0,255),2)


imgPatchA = A[yright:yright+patchsize,xright:xright+patchsize]
imgPatchB = B[yright:yright+patchsize,xright:xright+patchsize]
plt.subplot(1,2,1)
plt.imshow(imgPatchA)
plt.subplot(1,2,2)
plt.imshow(imgPatchB)
plt.show()