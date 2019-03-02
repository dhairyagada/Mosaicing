import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from numpy.linalg import inv
from mynetconfig import *
from glob import glob
import os
from pylab import *

def GetInputPatches(pA,pB):

    imgA = cv2.imread(pA)
    imgB = cv2.imread(pB)

    imgA = cv2.resize(imgA,(w,h))
    imgB = cv2.resize(imgB,(w,h))

    A = cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)
    B = cv2.cvtColor(imgB,cv2.COLOR_RGB2GRAY)

    A = A[topbottomcrop:h,w-sidecrop:w]
    B = B[0:h-topbottomcrop,0:sidecrop]

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
    """ A = cv2.polylines(A,points,1,(0,0,255),2)
    B = cv2.polylines(B,points,1,(0,0,255),2)
    plt.subplot(1,2,1)
    plt.imshow(A)
    plt.subplot(1,2,2)
    plt.imshow(B)
    plt.show() """

    imgPatchA = A[yright:yright+patchsize,xright:xright+patchsize]
    imgPatchB = B[yright:yright+patchsize,xright:xright+patchsize]

    inputPatches = np.dstack((imgPatchA,imgPatchB))
    inputPatches = inputPatches.reshape((1,128,128,2))
    """ plt.subplot(1,2,1)
    plt.imshow(imgPatchA)
    plt.subplot(1,2,2)
    plt.imshow(imgPatchB)
    plt.show() """
    return inputPatches,points

pathA = './ImageProc/InputImages/test1.jpg'
pathB = './ImageProc/InputImages/test2.jpg'

inputPatches,points = GetInputPatches(pathA,pathB)
print(inputPatches.shape)