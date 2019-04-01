import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from numpy.linalg import inv
from mynetconfig import *
from glob import glob
import os
from pylab import *

def show(im):
    plt.plot()
    plt.imshow(im)
    plt.show()
    return


def show2(imA,imB):
    plt.subplot(1,2,1)
    plt.imshow(imA)
    plt.subplot(1,2,2)
    plt.imshow(imB)
    plt.show()
    return
def GetInputPatches(pA,pB):

    imgA = cv2.imread(pA)
    imgB = cv2.imread(pB)

    imgA = cv2.resize(imgA,(w,h))
    imgB = cv2.resize(imgB,(w,h))

    image_width = imgA.shape[1]
    image_height = imgB.shape[0]

    gap = int((image_height-image_width)/2)

    leftpatch = imgA[gap:image_width+gap,0:image_width]
    rightpatch = imgB[gap:image_width+gap,0:image_width]

    leftpatch = cv2.cvtColor(leftpatch,cv2.COLOR_RGB2GRAY)
    rightpatch = cv2.cvtColor(rightpatch,cv2.COLOR_RGB2GRAY)

    leftpatch = cv2.resize(leftpatch,(128,128))
    rightpatch = cv2.resize(rightpatch,(128,128))

    upleft = (0,0)
    botleft = (w,0)
    botright = (w,w)
    upright = (0,w)

    points = [upleft,botleft,botright,upright]

    points = np.array(points)
    points = points.reshape((1,4,2))
   
    inputPatches = np.dstack((leftpatch,rightpatch))
    inputPatches = inputPatches.reshape((1,128,128,2))
    return inputPatches,points

""" pathA = './ImageProc/InputImages/cycle1.jpeg'
pathB = './ImageProc/InputImages/cycle2.jpeg'

inputPatches,points = GetInputPatches(pathA,pathB) """
#print(inputPatches.shape)