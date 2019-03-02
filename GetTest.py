import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from numpy.linalg import inv
pathA = './ImageProc/InputImages/home1.jpg'
pathB = './ImageProc/InputImages/home2.jpg'
imgA = cv2.imread(pathA)
imgB = cv2.imread(pathB)

from mynetconfig import *
from glob import glob
import os
from pylab import *

pathA = './ImageProc/InputImages/home1.jpg'
pathB = './ImageProc/InputImages/home2.jpg'
imgA = cv2.imread(pathA)
imgB = cv2.imread(pathB)

imgA = cv2.resize(imgA,(w,h))
imgB = cv2.resize(imgB,(w,h))

A = cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)
B = cv2.cvtColor(imgB,cv2.COLOR_RGB2GRAY)

A = imgA[0:h,w-270:w]
B = imgB[0:h,0:270]

plt.subplot(1,2,1)
plt.imshow(A)
plt.subplot(1,2,2)
plt.imshow(B)
plt.show()
xright = randint(rho,x_l-rho-patchsize)
yright = randint(rho,h-rho-patchsize-newpointdel)
cv2.rectangle(imgA,(rho,rho),(x_l-rho-patchsize,h-rho-patchsize),(0,255,255),2)


upleft = (xright,yright)
botleft = (xright,yright+patchsize)
botright = (xright+patchsize,yright+patchsize)
upright = (xright+patchsize,yright)

points = [upleft,botleft,botright,upright]

points = np.array(points)
points = points.reshape((1,4,2))
imgA = cv2.polylines(imgB,points,1,(0,0,255),2)


    